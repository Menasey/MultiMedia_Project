import os
import pickle
import logging
from datetime import datetime
from typing import List

import numpy as np
from fastapi import (
    FastAPI, HTTPException, Depends, File, UploadFile, Form,
    BackgroundTasks, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from backend.processor import (
    extract_urls, scrape_urls,
    build_features_and_vect,
    train_one_class_model,
    compute_threshold,
    score_sample,
)
from backend.visualization import (
    plot_decision_scores,
    plot_tfidf_term_importance,
    plot_fold_scores
)
from sqlalchemy.exc import SQLAlchemyError
from backend.evaluation import evaluate_model  # real evaluation

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📂 Expose saved_models folder
app.mount("/saved_models", StaticFiles(directory="saved_models"), name="saved_models")



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



@app.get("/models")
def list_models(db: Session = Depends(get_db)):
    models = db.query(TrainedModel).order_by(TrainedModel.training_date.desc()).all()
    return [
        {
            "id": m.id,
            "name": m.name,
            "classifier": m.classifier,
            "description": m.description, 
            "training_date": m.training_date,
            "eval_score_plot": m.eval_score_plot,
            "eval_terms_plot": m.eval_terms_plot,
            "eval_fold_plot": m.eval_fold_plot,
        }
        for m in models
    ]



def _do_training(
    job_id: int,
    zip_bytes: bytes,
    filename: str,
    classifier: str,
    model_name: str,
):
    db = SessionLocal()
    try:
        logger.info(f"[JOB {job_id}] Starting training job for {classifier}")
        job = db.query(TrainingJob).get(job_id)
        job.status = "running"
        db.commit()

        tmp_zip = f"tmp_{filename}"
        with open(tmp_zip, "wb") as fh:
            fh.write(zip_bytes)
        logger.info(f"[JOB {job_id}] Zip file written to {tmp_zip}")

        urls = extract_urls(tmp_zip)
        logger.info(f"[JOB {job_id}] Extracted {len(urls)} URLs from zip")
        os.remove(tmp_zip)

        texts: List[str] = scrape_urls(urls)
        logger.info(f"[JOB {job_id}] Scraped {len(texts)} texts")

        if not any(t.strip() for t in texts):
            raise RuntimeError("All scraped texts are empty")

        X_raw, vectorizer_full, top_terms = build_features_and_vect(texts)
        logger.info(f"[JOB {job_id}] Feature extraction complete")

        model, scores = train_one_class_model(
            classifier,
            X_raw,
            nu=0.1,
            gamma= 'auto',
            latent_dim=32,
            epochs=50,
            batch_size=16,
        )
        logger.info(f"[JOB {job_id}] Model training complete")

        #alpha = 1.5 if classifier == "deep_one_class" else 0.5

        threshold = compute_threshold(scores, alpha=0.5)
        logger.info(f"[JOB {job_id}] Threshold computed: {threshold}")

        y_true = np.ones(X_raw.shape[0])
        results = evaluate_model(X_raw, y_true, model, threshold=threshold)
        logger.info(f"[JOB {job_id}] Evaluation done")

        os.makedirs("saved_models", exist_ok=True)
        path = f"saved_models/{classifier}_{int(datetime.utcnow().timestamp())}.pkl"
        with open(path, "wb") as fh:
            pickle.dump({
                "model": model,
                "vect": vectorizer_full,
                "top_terms": top_terms,
                "threshold": threshold,
            }, fh)
        logger.info(f"[JOB {job_id}] Model artifacts saved")

        eval_score_path = path.replace(".pkl", "_score.png")
        eval_terms_path = path.replace(".pkl", "_terms.png")
        eval_fold_path = path.replace(".pkl", "_folds.png")

        plot_decision_scores(model, X_raw, outpath=eval_score_path)
        plot_tfidf_term_importance(X_raw, top_terms, outpath=eval_terms_path)
        plot_fold_scores(results, outpath=eval_fold_path)
        logger.info(f"[JOB {job_id}] All plots generated")

        db_model = TrainedModel(
            name=model_name,
            classifier=classifier,
            file_path=path,
            eval_score_plot=eval_score_path,
            eval_terms_plot=eval_terms_path,
            eval_fold_plot=eval_fold_path,
        )
        db.add(db_model)
        job.status = "completed"
        job.model_id = db_model.id
        db.commit()
        logger.info(f"[JOB {job_id}] SUCCESS ✅")

    except Exception as exc:
        logger.exception(f"[JOB {job_id}] FAILED ❌: {exc}")
        job.status = "failed"
        job.error = str(exc)
        db.commit()
    finally:
        db.close()



@app.post("/models/train", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_training(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    classifier: str = Form("iforest"),
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

# DELETE a single model (and its related collections + items)
@app.delete("/models/{model_id}", status_code=204)
def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_active_admin)
):
    model = db.query(TrainedModel).filter_by(id=model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Delete all collections using this model (and their items)
        collections = db.query(Collection).filter_by(model_id=model_id).all()
        for coll in collections:
            db.query(CollectionItem).filter_by(collection_id=coll.id).delete()
            db.delete(coll)

        # Delete the model itself
        db.delete(model)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")

    return {"detail": "Model deleted successfully"}

# DELETE all models (and related collections + items)
@app.delete("/models", status_code=204)
def delete_all_models(
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_active_admin)
):
    try:
        db.query(CollectionItem).delete()
        db.query(Collection).delete()
        db.query(TrainedModel).delete()
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete all models: {e}")

    return {"detail": "All models and related data deleted"}


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
#  COLLECTIONS
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

    tmp_path = f"tmpc_{file.filename}"
    with open(tmp_path, "wb") as fh:
        fh.write(await file.read())
    urls = extract_urls(tmp_path)
    os.remove(tmp_path)

    tm = db.query(TrainedModel).filter_by(id=coll.model_id).first()
    with open(tm.file_path, "rb") as f:
        artefacts = pickle.load(f)

    model = artefacts["model"]
    vect = artefacts["vect"]
    threshold = artefacts["threshold"]

    added = 0
    for url in urls:
        text = scrape_urls([url])[0]
        if not text.strip():
            logger.warning(f"⚠️ Empty content at {url}, skipping.")
            continue

        Xf = vect.transform([text])
        if Xf.nnz == 0:
            logger.warning(f"⚠️ TF-IDF vector is all zeros for {url}, skipping.")
            continue

        score = score_sample(model, Xf)
        logger.info(f"[Prediction] {url} → Score = {score:.6f}, Threshold = {threshold:.6f}")

        classifier = getattr(model, "__class__", type(model)).__name__.lower()
        if "deep" in classifier:
            # Deep One Class: Lower (more negative) = Crisis
            pred = "Crisis" if score <= threshold else "Non-Crisis"
        else:
            # SVM: Higher (more positive) = Crisis
            pred = "Crisis" if score >= threshold else "Non-Crisis"


        db.add(CollectionItem(
            collection_id=cid,
            url=url,
            prediction=pred,
            score=float(score),
        ))
        added += 1

    db.commit()
    return {"added": added}


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
    return {
        "id": coll.id,
        "title": coll.title,
        "model_id": coll.model_id,  
        "items": items,
    }

@app.delete("/collections/{cid}", status_code=204)
def delete_collection(
    cid: int,
    current=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    collection = db.query(Collection).filter_by(id=cid, user_id=current.id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    try:
        # Delete collection items first
        db.query(CollectionItem).filter_by(collection_id=cid).delete()
        db.delete(collection)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")

    return {"detail": "Collection deleted"}