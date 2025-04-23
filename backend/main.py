import os
import pickle
import logging
from datetime import datetime

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend import models, database, auth
from backend.auth import (
    get_db,
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_current_active_admin,
)
from backend.models import User, TrainingJob, TrainedModel, Collection, CollectionItem
from backend.processor import (
    extract_urls,
    scrape_urls,
    encode_texts_with_selected_terms,
    get_top_terms_by_tfidf,
    reduce_dimensionality,
    normalize_vectors,
    get_model,
)
from backend.database import SessionLocal, engine
from .deep_one_class import DeepOneClassClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM 
from eif import iForest as ExtendedIsolationForest

# ─── Logging Configuration ──────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize database tables
models.Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(title="Crisis Events Text Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── AUTH & REGISTRATION ───────────────────────────────────────────────────────
@app.post("/register")
def register(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),
    db: Session = Depends(get_db),
):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(400, "Username already taken")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already registered")

    user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    logger.info(f"[REGISTER] user={username!r} admin={is_admin}")
    return {"msg": "Registered successfully"}

@app.post("/token")
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning(f"[LOGIN] failure for user={form_data.username!r}")
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user.username})
    logger.info(f"[LOGIN] success for user={user.username!r}")
    return {"access_token": token, "token_type": "bearer", "is_admin": user.is_admin}

# ─── LIST MODELS ────────────────────────────────────────────────────────────────
@app.get("/models")
def list_models(db: Session = Depends(get_db)):
    models_list = (
        db.query(TrainedModel)
          .order_by(TrainedModel.training_date.desc())
          .all()
    )
    logger.info(f"[LIST_MODELS] returned {len(models_list)} models")
    return models_list

# ─── TRAINING → Asynchronous with BackgroundTasks ───────────────────────────────
def _do_training(
    job_id: int,
    zip_bytes: bytes,
    filename: str,
    classifier: str,
    model_name: str,
):
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).get(job_id)
        job.status = "running"
        job.updated_at = datetime.utcnow()
        db.commit()
        logger.info(f"[JOB {job_id}] started training '{classifier}' with name '{model_name}'")

        # Write and extract URLs
        tmp_path = f"tmp_{filename}"
        with open(tmp_path, "wb") as f:
            f.write(zip_bytes)
        urls = extract_urls(tmp_path)
        os.remove(tmp_path)
        logger.info(f"[JOB {job_id}] extracted {len(urls)} URLs")

        # Prepare features
        texts = scrape_urls(urls)
        X_full, vect_full = encode_texts_with_selected_terms(texts, [])
        top_terms = get_top_terms_by_tfidf(X_full, vect_full, n_terms=300)
        X, vect = encode_texts_with_selected_terms(texts, top_terms)
        Xr, svd = reduce_dimensionality(X, n_components=100)
        Xf = normalize_vectors(Xr)
        logger.info(f"[JOB {job_id}] data prepared: shape {Xf.shape}")

        if classifier == "deep_one_class":
            logger.info(f"[JOB {job_id}] Processing {len(texts)} texts for DeepOneClassClassifier")
            X_temp = X.toarray()
            input_dim = X_temp.shape[1]
            model = DeepOneClassClassifier(input_dim=input_dim, latent_dim=16, epochs=20, batch_size=32)
            model.fit(X_temp)
            logger.info(f"[JOB {job_id}] DeepOneClassClassifier trained successfully")
            training_scores = model.decision_function(X_temp)

        elif classifier == "eif":
            logger.info(f"[JOB {job_id}] Training Extended Isolation Forest")
            model = ExtendedIsolationForest(ntrees=50, sample_size=256, random_seed=42)
            model.fit(Xf)
            logger.info(f"[JOB {job_id}] Extended Isolation Forest trained successfully")
            threshold = None  # Skip threshold calculation for EIF

        else:
            model = get_model(classifier)
            model.fit(Xf)
            training_scores = model.decision_function(Xf)

        logger.info(f"[JOB {job_id}] {classifier} trained successfully")

        # Serialize
        os.makedirs("saved_models", exist_ok=True)
        fname = f"saved_models/{classifier}_{int(datetime.utcnow().timestamp())}.pkl"
        with open(fname, "wb") as f:
            pickle.dump({
                "model": model,
                "vect": vect,
                "svd": svd,
                "top_terms": top_terms,
                "threshold": (model.decision_function(Xf).mean() - 6 * model.decision_function(Xf).std())
                if classifier not in ["randomforest", "eif"] else None,
            }, f)
        logger.info(f"[JOB {job_id}] serialized to {fname}")

        # Persist metadata
        db_model = TrainedModel(
            name=model_name,
            classifier=classifier,
            file_path=fname,
        )
        db.add(db_model)
        db.commit()
        logger.info(f"[JOB {job_id}] TrainedModel id={db_model.id} created")

        # Mark job complete
        job.status = "completed"
        job.model_id = db_model.id
        job.updated_at = datetime.utcnow()
        db.commit()

    except Exception as exc:
        logger.exception(f"[JOB {job_id}] failed")
        job = db.query(TrainingJob).get(job_id)
        job.status = "failed"
        job.error = str(exc)
        job.updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()

@app.post("/models/train", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_training(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    classifier: str = Form("svm"),
    model_name: str = Form(...),
    current_admin=Depends(get_current_active_admin),
    db: Session = Depends(get_db),
):
    data = await zip_file.read()

    job = TrainingJob(
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    logger.info(f"[JOB {job.id}] enqueued by admin={current_admin.username!r}")
    background_tasks.add_task(
        _do_training,
        job.id,
        data,
        zip_file.filename,
        classifier,
        model_name,
    )
    return {"job_id": job.id}

@app.get("/training_jobs/{job_id}")
def get_job_status(
    job_id: int,
    current=Depends(get_current_active_admin),
    db: Session = Depends(get_db)
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
        "updated_at": job.updated_at
    }

# ─── COLLECTIONS & PREDICTION ──────────────────────────────────────────────────
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
    logger.info(f"[CREATE_COLL] user={current.username!r} coll_id={coll.id}")
    return coll

@app.get("/collections")
def list_collections(
    current=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    cols = db.query(Collection).filter(Collection.user_id == current.id).all()
    logger.info(f"[LIST_COLL] user={current.username!r} has {len(cols)} collections")
    return cols

@app.post("/collections/{cid}/predict")
async def predict_collection(
    cid: int,
    file: UploadFile = File(...),
    current=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    logger.info(f"[PREDICT] user={current.username!r} coll_id={cid} file={file.filename!r}")

    coll = db.query(Collection).filter_by(id=cid, user_id=current.id).first()
    if not coll:
        raise HTTPException(404, "Collection not found")

    tmp_path = f"tmpc_{file.filename}"
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    urls = extract_urls(tmp_path)
    os.remove(tmp_path)
    logger.info(f"[PREDICT] extracted {len(urls)} URLs")

    tm = db.query(TrainedModel).filter_by(id=coll.model_id).first()
    with open(tm.file_path, "rb") as f:
        data = pickle.load(f)
    model, vect, svd, threshold = data["model"], data["vect"], data["svd"], data.get("threshold")

    # Log the loaded model and threshold
    logger.info(f"[PREDICT] loaded model_id={tm.id!r} with threshold={threshold}")

    added = 0
    for i, url in enumerate(urls, 1):
        text = scrape_urls([url])[0]
        if not text.strip():
            logger.warning(f"[PREDICT] Skipping URL {url} due to empty content")
            continue

        X = vect.transform([text])
        Xr = svd.transform(X)
        Xf = normalize_vectors(Xr)

        if Xf.shape[0] == 0 or Xf.shape[1] == 0:
            logger.warning(f"[PREDICT] Skipping URL {url} due to empty or malformed Xf")
            continue

        try:
            if isinstance(model, ExtendedIsolationForest):
                score = model.compute_paths(Xf)[0]  # Use compute_paths for EIF
                pred = "Crisis" if score > 0.5 else "Non-Crisis"  # Adjust threshold as needed
            elif hasattr(model, "decision_function"):
                if threshold is None:
                    raise ValueError("Threshold not found for decision_function-based model")
                score = model.decision_function(Xf)[0]
                pred = "Crisis" if score >= threshold else "Non-Crisis"
            elif hasattr(model, "score_samples"):
                if threshold is None:
                    raise ValueError("Threshold not found for score_samples-based model")
                score = model.score_samples(Xf)[0]
                pred = "Crisis" if score >= threshold else "Non-Crisis"
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")
        except Exception as e:
            logger.exception(f"[PREDICT] Error during prediction for URL {url}: {e}")
            continue

        reconstruction_error = -score  # Negative because higher error = more anomalous
        logger.info(f"[PREDICT] URL={url} Reconstruction Error={reconstruction_error:.4f}")

        item = CollectionItem(
            collection_id=cid, url=url, prediction=pred, score=float(score)
        )
        db.add(item)
        added += 1

    db.commit()
    logger.info(f"[PREDICT] committed {added} items to DB")
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
    items = [{"url": i.url, "pred": i.prediction, "score": i.score} for i in coll.items]
    logger.info(f"[GET_COLL] returning {len(items)} items for collection {cid}")
    return {"id": coll.id, "title": coll.title, "items": items}