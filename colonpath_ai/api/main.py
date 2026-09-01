"""
FastAPI Main Application Entry Point for COLONPATH-AI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.routes import health, analysis, cases, regions, review, copilot

app = FastAPI(
    title="COLONPATH-AI Decision Support API",
    description="Multimodal GI Foundation Model & Decision Support System for Colorectal Histopathology",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for mobile & web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(analysis.router)
app.include_router(cases.router)
app.include_router(regions.router)
app.include_router(regions.alias_router)
app.include_router(review.router)
app.include_router(copilot.router)

# Mount outputs directory for static access if needed
outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(outputs_dir)), name="static")


from fastapi.responses import FileResponse

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

@app.get("/")
@app.get("/viewer")
def root_dashboard():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "system": "COLONPATH-AI",
        "description": "Multimodal GI Foundation Decision Support API",
        "documentation": "/docs",
        "health": "/health",
    }
