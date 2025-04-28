# ==============================================
#  main.py  —  FastAPI 
# ==============================================
"""
Light-weight orchestration only:
• heavy feature / ML logic now lives in backend.processor
• all original endpoints kept intact (no breaking changes)
"""

import os
import pickle
import logging
from datetime import datetime
from typing import List

import numpy as np
from fastapi import (
    FastAPI, HTTPException, Depends, File, UploadFile, Form,
    BackgroundTasks, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

# ── project-local imports ────────────────────────────────────────────────────
from backend import models
from backend.auth import (
    get_db, get_password_hash, authenticate_user, create_access_token,
    get_current_active_user, get_current_active_admin,
)
from backend.models import (
    User, TrainingJob, TrainedModel,
    Collection, CollectionItem,
)
from backend.database import SessionLocal, engine
from backend.processor import (                 # new helper API
    extract_urls, scrape_urls,
    build_features_and_vect,        # TF-IDF → SVD → L2  (returns X, vect, terms)
    train_one_class_model,          # fits model & returns (model, scores)
    compute_threshold,              # 95 % inlier cut-off
    score_sample,                   # normality score for one sample
)
# deep OC is still imported inside processor; none needed here

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App & DB bootstrap ──────────────────────────────────────────────────────
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Crisis Events Text Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
#  AUTH  (unchanged)
# ============================================================================

@app.post("/register")
def register(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),
    db: Session = Depends(get_db),
):
    if db.query(User).filter_by(username=username).first():
        raise HTTPException(400, "Username already taken")
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(400, "Email already registered")

    db.add(User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
        is_admin=is_admin,
    ))
    db.commit()
    return {"msg": "Registered successfully"}


@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(401, "Incorrect username or password")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "is_admin": user.is_admin}

# ============================================================================
#  MODEL CATALOGUE  (unchanged)
# ============================================================================

@app.get("/models")
def list_models(db: Session = Depends(get_db)):
    return db.query(TrainedModel).order_by(TrainedModel.training_date.desc()).all()

# ============================================================================
#  BACKGROUND TRAINING TASK
# ============================================================================

def _do_training(
    job_id: int,
    zip_bytes: bytes,
    filename: str,
    classifier: str,
    model_name: str,
):
    """
    Heavy work is delegated to backend.processor helpers.
    """
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).get(job_id)
        job.status = "running"
        db.commit()

        # 1️⃣  unzip → URLs → scrape
        tmp_zip = f"tmp_{filename}"
        with open(tmp_zip, "wb") as fh:
            fh.write(zip_bytes)
        urls = extract_urls(tmp_zip)
        os.remove(tmp_zip)

        texts: List[str] = scrape_urls(urls)
        if not any(t.strip() for t in texts):
            raise RuntimeError("All scraped texts are empty")

        # 2️⃣  Build feature matrix & vectoriser
        Xf, vect, top_terms = build_features_and_vect(texts)

        # 3️⃣  Train model & get training scores
        model, scores = train_one_class_model(classifier, Xf)

        # 4️⃣  Threshold for all models
        threshold = compute_threshold(scores)

        # 5️⃣  Save artefacts
        os.makedirs("saved_models", exist_ok=True)
        path = f"saved_models/{classifier}_{int(datetime.utcnow().timestamp())}.pkl"
        with open(path, "wb") as fh:
            pickle.dump({
                "model": model,
                "vect": vect,
                "top_terms": top_terms,
                "threshold": threshold,
            }, fh)

        # 6️⃣  DB bookkeeping
        db_model = TrainedModel(
            name=model_name,
            classifier=classifier,
            file_path=path
        )
        db.add(db_model)
        job.status = "completed"
        job.model_id = db_model.id
        db.commit()
        logger.info(f"[JOB {job_id}] {classifier} trained OK")

    except Exception as exc:
        logger.exception(f"[JOB {job_id}] failed: {exc}")
        job.status = "failed"
        job.error = str(exc)
        db.commit()
    finally:
        db.close()

# ============================================================================
#  TRAINING ENQUEUE ENDPOINT
# ============================================================================

@app.post("/models/train", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_training(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    classifier: str = Form("iforest"),      # default
    model_name: str = Form(...),
    current_admin=Depends(get_current_active_admin),
    db: Session = Depends(get_db),
):
    data = await zip_file.read()
    job = TrainingJob(
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(
        _do_training, job.id, data, zip_file.filename,
        classifier.lower(), model_name
    )
    return {"job_id": job.id}

# ============================================================================
#  TRAINING-JOB STATUS  (unchanged)
# ============================================================================

@app.get("/training_jobs/{job_id}")
def get_job_status(
    job_id: int,
    current=Depends(get_current_active_admin),
    db: Session = Depends(get_db),
):
    job = db.query(TrainingJob).get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "model_id": job.model_id,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }

# ============================================================================
#  COLLECTIONS  (list / create)  — unchanged
# ============================================================================

@app.get("/collections")
def list_collections(
    current=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    return db.query(Collection).filter_by(user_id=current.id).all()


@app.post("/collections")
def create_collection(
    title: str = Form(...),
    model_id: int = Form(...),
    current=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    coll = Collection(title=title, user_id=current.id, model_id=model_id)
    db.add(coll)
    db.commit()
    db.refresh(coll)
    return coll

# ============================================================================
#  PREDICT ENDPOINT  (uses processor.score_sample)
# ============================================================================

@app.post("/collections/{cid}/predict")
async def predict_collection(
    cid: int,
    file: UploadFile = File(...),
    current=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    coll = db.query(Collection).filter_by(id=cid, user_id=current.id).first()
    if not coll:
        raise HTTPException(404, "Collection not found")

    # unzip URLs to predict
    tmp_path = f"tmpc_{file.filename}"
    with open(tmp_path, "wb") as fh:
        fh.write(await file.read())
    urls = extract_urls(tmp_path)
    os.remove(tmp_path)

    # load model artefacts
    tm = db.query(TrainedModel).filter_by(id=coll.model_id).first()
    with open(tm.file_path, "rb") as f:
        artefacts = pickle.load(f)
    model, vect, threshold = (
        artefacts["model"],
        artefacts["vect"],
        artefacts["threshold"],
    )

    added = 0
    for url in urls:
        text = scrape_urls([url])[0]
        if not text.strip():
            continue

        Xf = vect.transform([text])
        score  = score_sample(model, Xf)
        pred   = "Crisis" if score >= threshold else "Non-Crisis"

        db.add(CollectionItem(
            collection_id=cid,
            url=url,
            prediction=pred,
            score=float(score),
        ))
        added += 1

    db.commit()
    return {"added": added}

# ============================================================================
#  GET COLLECTION  (unchanged)
# ============================================================================

@app.get("/collections/{cid}")
def get_collection(
    cid: int,
    current=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    coll = db.query(Collection).filter_by(id=cid, user_id=current.id).first()
    if not coll:
        raise HTTPException(404, "Collection not found")
    items = [
        {"url": it.url, "pred": it.prediction, "score": it.score}
        for it in coll.items
    ]
    return {"id": coll.id, "title": coll.title, "items": items}
