# HeartSense 🫀

**Cardiovascular Disease Prediction & Dynamic Risk Assessment System**

A modern, full-stack machine learning application built with React, FastAPI, Scikit-learn, and PostgreSQL. It implements a rigorous data science pipeline and a dynamic temporal risk-assessment engine for patient visits.

---

## 1. Project Overview
HeartSense is a Cardiovascular Disease Prediction System. The application evaluates patient clinical information and provides a cardiovascular risk assessment using a robust machine-learning backend.

## 2. Objectives
- Provide a clear, elderly-friendly user interface for cardiovascular risk assessment.
- Implement a strict, reproducible machine learning pipeline.
- Offer dynamic temporal risk assessment comparing historical clinical data.
- Ensure clean architectural separation between frontend (React) and backend (FastAPI/Python).

## 3. System Architecture
The application uses a decoupled architecture:
- **Frontend**: React (TypeScript, Vite) providing a clean, responsive medical dashboard.
- **Backend**: Python (FastAPI) serving the REST API and orchestrating the ML pipeline.
- **Database**: PostgreSQL (via SQLAlchemy) storing patient histories and assessment results.

## 4. Phase 1 – Data Preparation
Exact required order implemented:
1. Patient Clinical Data (Cleveland Heart Disease Dataset)
2. Data Preprocessing (Imputation, numerical conversion)
3. **SMOTE** (Class balancing applied *before* train-test split)
4. Feature Selection (SelectKBest applied *after* SMOTE)
5. Train-Test Split (Stratified split creating final Training and Test datasets)

## 5. Phase 2 – Model Development & Selection
Uses exactly these 6 models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Gaussian Naive Bayes

Hyperparameter tuning is performed via `GridSearchCV` exclusively on the training set. The best-performing model is automatically selected based on evaluation metrics.

## 6. Phase 3 – Model Evaluation
Models are evaluated using:
- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-Score
- ROC-AUC

## 7. Phase 4 – Dynamic / Temporal Risk Assessment
Tracks patient clinical values across multiple chronological visits.
- Retrieves historical records
- Uses timestamps
- Compares previous and current clinical values
- Identifies relevant changes (Increased, Decreased, Stable)
- Generates current risk assessment

## 8. ML Models
The backend strictly uses the 6 approved models mentioned in Phase 2. No deep learning or additional gradient boosting frameworks (like XGBoost or LightGBM) are used.

## 9. Evaluation Metrics
Metrics are generated during the training phase and served via the `/api/analytics` endpoint for visualization in the React dashboard.

## 10. Frontend Architecture
- **React + TypeScript + Vite**
- Uses `react-router-dom` for routing (`/`, `/assessment`, `/history`, `/analytics`, `/about`).
- Modern, clean, elderly-friendly dashboard with Lucide icons.
- Separated components (`Sidebar.tsx`, `AssessmentForm.tsx`, `ReportUpload.tsx`, etc.).

## 11. Backend Architecture
- **Python + FastAPI**
- RESTful endpoints bridging the ML logic and the PostgreSQL database.
- Uses Pydantic schemas for strict data validation matching the ML feature requirements.

## 12. Database
- **PostgreSQL** configured via `.env`.
- Stores patient information, clinical values, prediction results, prediction probabilities, models used, and timestamps.

## 13. API Endpoints
- `POST /api/assessment/predict`: Submits clinical data and returns risk prediction.
- `GET /api/history/{patient_code}`: Retrieves chronological assessment history.
- `GET /api/analytics`: Retrieves model evaluation metrics.

## 14. Installation
Ensure you have Python 3.10+, Node.js (v18+), and PostgreSQL installed.

### Database Setup
Create a `.env` file in the `backend/` directory:
```text
DATABASE_URL=postgresql://username:password@localhost:5432/heartsense
```

## 15. Running the Frontend
```bash
cd frontend
npm install
npm run dev
```
The frontend will run at `http://localhost:5173` (or `5174`).

## 16. Running the Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r ..\requirements.txt
uvicorn app.main:app --reload
```
The API will run at `http://127.0.0.1:8000`.

## 17. Project Folder Structure
```text
HeartSense/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── models.py
│   │   └── database.py
│   ├── src/             # ML Pipeline logic
│   ├── artifacts/       # Saved models & metrics
│   ├── data/            # Datasets
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── components/  # Reusable React components
│   │   ├── pages/       # Route components
│   │   ├── services/    # API calls (axios)
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

## 18. Limitations
- Automatic medical report extraction (OCR) is not yet fully implemented; manual entry of clinical values is required.
- Incremental Learning and Cross-Dataset Validation are out-of-scope for the current implementation.

## 19. Disclaimer
This assessment tool is for research and decision-support purposes only. It is **not** a medical diagnosis. Consult a qualified healthcare professional for medical advice.
