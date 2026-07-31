"""
Pipeline complet d'analyse de sentiment vocal
Orchestre : Audio Processing -- ASR (Wav2Vec) -- Sentiment Analysis (BERT)
"""

from typing import Dict, Union
from pathlib import Path
import numpy as np
import soundfile as sf
from io import BytesIO
import logging

from .audio_processor import AudioProcessor
from .asr_model import ASRModel
from .sentiment_model import SentimentAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalysisPipeline:
    """Pipeline complet de bout en bout"""
    
    def __init__(self, device: str = None):
        """
        Initialise tous les composants du pipeline
        
        Args:
            device: 'cpu' ou 'cuda'. Auto-détection si None.
        """
        logger.info("Initialisation du pipeline complet")
        
        self.audio_processor = AudioProcessor()
        self.asr_model = ASRModel(device=device)
        
        # Le sentiment analyzer prend device en int (-1 pour CPU, 0+ pour GPU)
        device_int = -1
        if device and device != "cpu":
            device_int = 0  # GPU
        self.sentiment_analyzer = SentimentAnalyzer(device=device_int)
        
        logger.info("Pipeline initialisé")
    
    def process(self, audio_path: str) -> Dict:
        """
        Traite un fichier audio complet de bout en bout
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Dict avec {
                'transcription': str,
                'sentiment': str (POSITIVE/NEGATIVE/NEUTRAL),
                'confidence': float,
                'models_used': dict,
                'duration': float (secondes)
            }
        """
        logger.info(f"Traitement du fichier: {audio_path}")
        
        try:
            # Étape 1 : Charger et prétraiter l'audio
            logger.info("Étape 1/3: Chargement et prétraitement audio")
            audio = self.audio_processor.process_file(audio_path)
            duration = len(audio) / self.audio_processor.TARGET_SR
            
            # Étape 2 : Transcription (ASR)
            logger.info("Étape 2/3: Transcription audio → texte (Wav2Vec 2.0)")
            transcription = self.asr_model.transcribe(audio)
            
            # Étape 3 : Analyse de sentiment
            logger.info("Étape 3/3: Analyse de sentiment (BERT)")
            sentiment_result = self.sentiment_analyzer.analyze(transcription)
            
            # Préparer la réponse
            result = {
                'transcription': transcription,
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'score': sentiment_result['score'],
                'duration': round(duration, 2),
                'models_used': {
                    'asr': self.asr_model.get_model_info(),
                    'sentiment': self.sentiment_analyzer.get_model_info()
                }
            }
            
            logger.info(f"Pipeline complété avec succès")
            logger.info(f"  - Sentiment: {result['sentiment']} ({result['confidence']:.2%})")
            logger.info(f"  - Durée: {result['duration']}s")
            
            return result
        
        except Exception as e:
            logger.error(f"Erreur dans le pipeline: {e}")
            raise
    
    def process_from_bytes(self, audio_bytes: bytes, format: str = "wav") -> Dict:
        """
        Traite des bytes audio directement
        Utile pour l'API et Gradio
        
        Args:
            audio_bytes: Contenu du fichier audio en bytes
            format: Format audio ('wav', 'mp3', etc.)
            
        Returns:
            Résultat du traitement complet
        """
        logger.info(f"Traitement de {len(audio_bytes)} bytes ({format})")
        
        try:
            # Sauvegarder temporairement les bytes dans un fichier temp
            import tempfile
            
            with tempfile.NamedTemporaryFile(
                suffix=f".{format}", 
                delete=False
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # Traiter le fichier temporaire
            result = self.process(tmp_path)
            
            # Nettoyer
            Path(tmp_path).unlink()
            
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement bytes: {e}")
            raise
    
    def process_multiple(self, audio_paths: list) -> list:
        """
        Traite plusieurs fichiers audio
        
        Args:
            audio_paths: Liste des chemins vers les fichiers audio
            
        Returns:
            Liste des résultats
        """
        results = []
        
        for i, audio_path in enumerate(audio_paths, 1):
            logger.info(f"Traitement {i}/{len(audio_paths)}")
            try:
                result = self.process(audio_path)
                results.append(result)
            except Exception as e:
                logger.warning(f"Erreur pour {audio_path}: {e}")
                results.append({
                    'file': audio_path,
                    'error': str(e)
                })
        
        return results
