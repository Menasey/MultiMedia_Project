# 🚨 Crisis Classifier

Crisis Classifier is a full-stack web application that allows administrators to **train machine learning models** on labeled crisis URL datasets and enables users to **classify new webpages** as *Crisis* or *Non-Crisis*. It includes rich visualizations, progress indicators, model management, and user-friendly interactions.

---

## 🧩 Features

### 👨‍💼 Admin Panel
- Upload labeled `.zip` datasets to train new models.
- Choose from multiple ML classifiers:
  - Support Vector Machine (SVM)
  - Isolation Forest (IForest)
  - Extended Isolation Forest (EIF)
  - Deep One-Class Autoencoder
- Add descriptive summaries to each model for clarity.
- Real-time training status with loading spinner and polling.
- Automatically view:
  - Decision Score Distributions
  - Top TF-IDF Terms
  - Cross-validation metrics

### 👤 User Panel
- Upload `.zip` files containing URLs to classify.
- Select from trained models with a visible description.
- Get classification results instantly per URL.
- Spinner and success feedback during collection creation.
- View visualizations of the selected model directly from the dashboard.

---

## 🛠️ Tech Stack

- **Frontend**: React + React Router + Tailwind CSS
- **Backend**: FastAPI (Python), Uvicorn
- **ML & Visualization**: scikit-learn, Matplotlib, pandas, NumPy
- **Authentication**: JWT-based user and admin login

---

## 📁 Project Structure

```
📁 backend/
│   ├── main.py                  # FastAPI app entry
│   ├── models.py                # SQLAlchemy models
│   ├── routes/                  # API endpoints
│   ├── jobs/                    # Model training and async processing
│   └── static/                  # Saved plots and files

📁 frontend/
│   ├── src/
│   │   ├── pages/               # AdminDashboard, UserDashboard, etc.
│   │   ├── services/api.js      # Axios service for backend interaction
│   │   └── global.css           # Tailwind + custom styles
│   └── main.jsx                 # App entry point
```

---

## 🚀 How to Run the App

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/crisis-classifier.git
cd crisis-classifier
```

### 2. Backend Setup (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload
```

The backend runs at: `http://127.0.0.1:8000`

### 3. Frontend Setup (React)
```bash
cd frontend
npm install
npm run dev
```

The frontend runs at: `http://localhost:5173`

---

## 🔒 Authentication Notes

- Users can **register** as either a regular user or admin.
- Admins have access to model training and management.
- JWT tokens are stored in `localStorage` and automatically attached to requests using Axios interceptors.

---

## 🧠 Future Enhancements

- Key-word & Entity Recognition
- Batch Scheduling
- Dockerized Deployment
- Basic URL Pre-Validation
- UI Refinements
- Better Collections
- Threshold Tuning Slider
- User Feedback Loop
- Lightweight Analytics Dashboard
- Admin Logs Page
- Multi-Language Stopword Support

---
