"""
Module de traitement audio
- Chargement des fichiers audio (.wav, .mp3)
- Rééchantillonnage à 16 kHz
- Normalisation
- Conversion en mono
"""

import soundfile as sf
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Processeur audio pour prétraitement"""
    
    TARGET_SR = 16000  # Sample rate cible (16 kHz pour Wav2Vec 2.0)
    MAX_DURATION = 5 * 60  # Durée maximale : 5 minutes
    SILENCE_THRESHOLD = 0.01  # Seuil de silence
    
    def __init__(self):
        """Initialise le processeur audio"""
        logger.info(f"AudioProcessor initialisé (target SR: {self.TARGET_SR} Hz)")
    
    def load_audio(self, audio_path: str) -> tuple:
        """
        Charge un fichier audio et le prétraite
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Tuple (audio_array, sample_rate)
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le format n'est pas supporté
        """
        audio_path = Path(audio_path)
        
        # Vérifier que le fichier existe
        if not audio_path.exists():
            raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
        
        # Vérifier le format supporté
        supported_formats = {'.wav', '.mp3', '.m4a', '.flac'}
        if audio_path.suffix.lower() not in supported_formats:
            raise ValueError(f"Format non supporté: {audio_path.suffix}. "
                           f"Formats acceptés: {supported_formats}")
        
        try:
            logger.info(f"Chargement du fichier: {audio_path}")
            
            # Charger avec soundfile (évite le problème pkg_resources de librosa)
            audio, sr = sf.read(str(audio_path))
            
            # Convertir en mono si stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            logger.info(f"Audio chargé: durée={len(audio)/sr:.2f}s, "
                       f"SR={sr}Hz")
            
            return audio, sr
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise
    
    def preprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Prétraite l'audio : rééchantillonnage, normalisation, mono
        
        Args:
            audio: Array audio
            sr: Sample rate original
            
        Returns:
            Audio prétraité (16 kHz, mono, normalisé)
        """
        logger.info(f"Prétraitement audio (SR original: {sr}Hz)")
        
        # 1. Vérifier la durée
        duration = len(audio) / sr
        if duration > self.MAX_DURATION:
            logger.warning(f"Audio trop long ({duration:.2f}s > {self.MAX_DURATION}s). "
                          f"Troncature...")
            max_samples = int(self.MAX_DURATION * sr)
            audio = audio[:max_samples]
        
        # 2. Rééchantillonner à 16 kHz si nécessaire
        if sr != self.TARGET_SR:
            logger.info(f"Rééchantillonnage: {sr}Hz → {self.TARGET_SR}Hz")
            # Utiliser soundfile avec resampling
            import scipy.signal
            num_samples = int(len(audio) * self.TARGET_SR / sr)
            audio = scipy.signal.resample(audio, num_samples)
        
        # 3. Normaliser l'amplitude
        audio = self.normalize_amplitude(audio)
        
        # 4. Vérifier le silence
        if self._is_silent(audio):
            logger.warning("Audio silencieux détecté")
        
        logger.info(f"Prétraitement complété: durée={len(audio)/self.TARGET_SR:.2f}s")
        
        return audio
    
    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalise l'amplitude de l'audio
        
        Args:
            audio: Array audio
            
        Returns:
            Audio normalisé ([-1, 1])
        """
        # Normaliser à [-1, 1] basé sur la valeur max absolue
        max_amplitude = np.max(np.abs(audio))
        
        if max_amplitude > 0:
            audio = audio / max_amplitude
        
        return audio.astype(np.float32)
    
    def _is_silent(self, audio: np.ndarray) -> bool:
        """
        Vérifie si l'audio est silencieux
        
        Args:
            audio: Array audio
            
        Returns:
            True si silencieux, False sinon
        """
        rms = np.sqrt(np.mean(audio ** 2))
        return rms < self.SILENCE_THRESHOLD
    
    def process_file(self, audio_path: str) -> np.ndarray:
        """
        Chargement + prétraitement en une seule fonction
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Audio prétraité
        """
        audio, sr = self.load_audio(audio_path)
        audio = self.preprocess(audio, sr)
        return audio