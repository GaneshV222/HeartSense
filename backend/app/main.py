from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import assessment, history, analytics
from app.database import init_db

app = FastAPI(title="HeartSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(assessment.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
