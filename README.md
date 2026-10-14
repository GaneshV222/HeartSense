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

## 4. Verified datasets and model training

The training workflow reads `backend/data/heart_disease_prediction_2026.csv` for labeled visits and `backend/data/temporal_dataset.xlsx` (worksheet `temporal_dataset`) for comparison only. Visit-derived history is authoritative; the workbook's precomputed previous values are not used as genuine history.

The model uses the 24 configured clinical features plus eight numeric changes and two lifestyle-change indicators. Patient IDs and labels are not predictors. SMOTENC is applied to the complete 100001-row labeled dataset before the train, validation, and test split, producing 140002 complete model records. The workflow compares the current XGBoost configuration with tuned XGBoost variants and a Gradient Boosting candidate; it reports measured metrics and does not guarantee a target accuracy.

Training without database replacement:

```powershell
python train.py
```

To explicitly import/replace source-scoped rows in PostgreSQL, supply the verified pre-import backup and exact database name:

```powershell
python train.py --import-database --backup-path "C:\path\to\verified-backup.csv" --expected-database heartsense
```

To store only the complete real-plus-synthetic SMOTE training records in PostgreSQL without replacing clinical visit rows:

```powershell
python train.py --store-smote-database --expected-database heartsense
```

The import checks that the backup patient IDs exactly match the dataset IDs and refuses to replace legacy patients with dependent application data. It does not drop tables or reset the database. Genuine visits remain in the clinical visit tables; the full real-plus-synthetic SMOTE result is stored separately in `smote_training_records` and exported to `backend/artifacts/smote_training_records.csv`. A validated model candidate is promoted only after training, artifact reload checks, and any requested import succeed.

## 5. Temporal risk assessment

Temporal changes are calculated from the immediately preceding genuine visit for the same patient. A first visit has null history-derived changes; missing history is not converted to zero. Rate-of-change features are not generated or used by the model. The existing PostgreSQL patient, visit, temporal-data, and prediction-history tables remain in use.

## 6. Evaluation

Training reports accuracy, precision, sensitivity, specificity, F1, ROC-AUC, false-positive/negative rates, per-class metrics, confusion matrices, stratified cross-validation on the resampled training matrix, and the original/resampled class counts. Results are written to `backend/artifacts/temporal_pipeline_summary.json` and `backend/artifacts/model_metrics.json`, and served through the existing analytics endpoint.

Relative `MODEL_ARTIFACT_DIR` values are resolved from `backend/`, independent of the command's working directory. The active temporal model and its feature manifest are stored together in that directory.

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
- Cross-Dataset Validation is out-of-scope for the current implementation.

## 19. Disclaimer
This assessment tool is for research and decision-support purposes only. It is **not** a medical diagnosis. Consult a qualified healthcare professional for medical advice.
