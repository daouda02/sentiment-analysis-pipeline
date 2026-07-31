"""
Module ASR (Automatic Speech Recognition)
Utilise Wav2Vec 2.0 pour la transcription audio en texte (français)
"""

import numpy as np
from transformers import AutoProcessor, AutoModelForCTC
import torch
import logging

logger = logging.getLogger(__name__)


class ASRModel:
    """Pipeline ASR avec Wav2Vec 2.0 pour le français"""
    
    MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    TARGET_SR = 16000
    
    def __init__(self, device: str = None):
        """
        Initialise le modèle Wav2Vec 2.0 pour le français
        
        Args:
            device: 'cpu' ou 'cuda'. Auto-détection si None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialisation ASRModel (device: {self.device})")
        
        self.processor = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle et le processeur depuis Hugging Face"""
        try:
            logger.info(f"Chargement du modèle: {self.MODEL_NAME}")
            
            # Charger le processeur SANS language model (évite problèmes d'encodage Windows)
            # Utiliser use_auth_token=False pour éviter les problèmes de cache
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.MODEL_NAME,
                    use_auth_token=False  # ← NOUVEAU
                )
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du processeur: {e}")
                logger.info("Tentative de chargement du processeur simple...")
                # Fallback: charger juste le processeur sans LM
                from transformers import Wav2Vec2Processor 
                self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_NAME)
            
            # Charger le modèle
            self.model = AutoModelForCTC.from_pretrained(
                self.MODEL_NAME,
                use_auth_token=False
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(" Modèle ASR chargé avec succès")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def transcribe(self, audio: np.ndarray, sr: int = None) -> str:
        """
        Transcrit un fichier audio en texte
        
        Args:
            audio: Array audio (doit être à 16 kHz et normalisé)
            sr: Sample rate (optionnel, assume 16kHz par défaut)
            
        Returns:
            Texte transcrit
        """
        if sr is None:
            sr = self.TARGET_SR
        
        logger.info(f"Transcription en cours... (durée: {len(audio)/sr:.2f}s)")
        
        try:
            with torch.no_grad():
                # Préparer les inputs
                inputs = self.processor(
                    audio, 
                    sampling_rate=sr, 
                    return_tensors="pt"
                )
                
                # Déplacer sur le bon device
                input_values = inputs["input_values"].to(self.device)
                
                # Inférence
                logits = self.model(input_values).logits
                
                # Décoder
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                
                logger.info(f"Transcription réussie")
                logger.debug(f"   Texte: {transcription[:100]}...")
                
                return transcription
        
        except Exception as e:
            logger.error(f"Erreur lors de la transcription: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Retourne les informations sur le modèle"""
        return {
            "model_name": self.MODEL_NAME,
            "device": self.device,
            "framework": "PyTorch",
            "task": "Automatic Speech Recognition (French)"
        }
