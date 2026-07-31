"""
Module Sentiment Analysis
Utilise un modèle BERT pré-fine-tuné pour la classification de sentiment
(Positif / Négatif / Neutre)
"""

from transformers import pipeline
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyseur de sentiment avec BERT multilingual"""
    
    MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    # Mapping des labels du modèle
    LABEL_MAPPING = {
        "1 star": "NEGATIVE",
        "2 stars": "NEGATIVE",
        "3 stars": "NEUTRAL",
        "4 stars": "POSITIVE",
        "5 stars": "POSITIVE",
    }
    
    def __init__(self, device: int = -1):
        """
        Initialise le modèle BERT pour l'analyse de sentiment
        
        Args:
            device: -1 pour CPU, 0+ pour GPU
        """
        self.device = device
        self.pipeline = None
        logger.info(f"Initialisation SentimentAnalyzer (device: {device})")
        
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle depuis Hugging Face"""
        try:
            logger.info(f"Chargement du modèle: {self.MODEL_NAME}")
            
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                device=self.device
            )
            
            logger.info("Modèle sentiment chargé avec succès")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def analyze(self, text: str) -> Dict:
        """
        Analyse le sentiment d'un texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dict avec {
                'sentiment': str (POSITIVE/NEGATIVE/NEUTRAL),
                'score': float (0-1),
                'label': str
            }
        """
        if not text or not text.strip():
            logger.warning("Texte vide fourni")
            return {
                'sentiment': 'NEUTRAL',
                'score': 0.0,
                'label': 'Texte vide'
            }
        
        try:
            logger.debug(f"Analyse de sentiment: {text[:50]}...")
            
            # Inférence
            results = self.pipeline(text)
            
            if not results or len(results) == 0:
                raise ValueError("Aucun résultat retourné par le modèle")
            
            # Extraire le premier résultat
            result = results[0]
            raw_label = result['label']
            score = result['score']
            
            # Mapper le label du modèle à nos classes
            sentiment = self.LABEL_MAPPING.get(raw_label, 'NEUTRAL')
            
            logger.debug(f"Résultat: {sentiment} (score: {score:.4f})")
            
            return {
                'sentiment': sentiment,
                'score': float(score),
                'label': raw_label,
                'confidence': float(score)
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Retourne les informations sur le modèle"""
        return {
            "model_name": self.MODEL_NAME,
            "device": self.device,
            "framework": "PyTorch",
            "task": "Sentiment Classification (Multilingual)",
            "classes": ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        }
