"""
API REST FastAPI
Endpoint POST /predict pour l'analyse de sentiment vocal
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path
import logging

# Ajouter le dossier src au chemin
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import SentimentAnalysisPipeline
from src.utils import validate_audio_file

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Modèles Pydantic pour la documentation Swagger
class SentimentResponse(BaseModel):
    """Réponse de prédiction"""
    transcription: str
    sentiment: str
    confidence: float
    duration: float
    models_used: dict


class ErrorResponse(BaseModel):
    """Réponse d'erreur"""
    error: str
    detail: str


# Initialiser l'app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API pour l'analyse de sentiment dans des appels vocaux français",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Ajouter CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser le pipeline au démarrage
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialise le pipeline au démarrage de l'API"""
    global pipeline
    try:
        logger.info("Démarrage de l'API...")
        pipeline = SentimentAnalysisPipeline(device="cpu")
        logger.info("Pipeline chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur au démarrage: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Nettoie les ressources à l'arrêt"""
    logger.info("Arrêt de l'API")


@app.get("/")
def read_root():
    """Endpoint racine - informations sur l'API"""
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "description": "Analyse de sentiment pour appels vocaux français",
        "endpoints": {
            "POST /predict": "Analyser un fichier audio",
            "GET /health": "Vérifier le statut de l'API",
            "GET /docs": "Documentation interactive Swagger",
        }
    }


@app.get("/health")
def health_check():
    """Vérifier le statut de l'API"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }


@app.post("/predict", response_model=SentimentResponse)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint principal pour prédire le sentiment d'un fichier audio
    
    **Paramètres:**
    - file: Fichier audio (.wav, .mp3, etc.)
    
    **Retourne:**
    ```json
    {
        "transcription": "Je suis très satisfait",
        "sentiment": "POSITIVE",
        "confidence": 0.95,
        "duration": 3.2,
        "models_used": {
            "asr": {...},
            "sentiment": {...}
        }
    }
    ```
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline non initialisé"
        )
    
    try:
        # Vérifier le type de fichier
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Nom de fichier manquant"
            )
        
        supported_formats = {'.wav', '.mp3', '.m4a', '.flac'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté: {file_ext}. "
                       f"Formats acceptés: {supported_formats}"
            )
        
        # Lire le contenu du fichier
        logger.info(f"Traitement du fichier: {file.filename}")
        content = await file.read()
        
        # Traiter via le pipeline
        result = pipeline.process_from_bytes(content, format=file_ext[1:])
        
        logger.info(f"Prédiction réussie: {result['sentiment']}")
        
        return SentimentResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )


@app.get("/models-info")
def get_models_info():
    """Retourne les informations sur les modèles utilisés"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline non initialisé")
    
    return {
        "asr_model": pipeline.asr_model.get_model_info(),
        "sentiment_model": pipeline.sentiment_analyzer.get_model_info()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
