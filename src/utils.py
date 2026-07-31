"""
Fonctions utilitaires
- Validation des fichiers
- Gestion des erreurs
- Logging
- Manipulations audio
"""

import logging
from pathlib import Path
import numpy as np
import librosa


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def validate_audio_file(audio_path: str) -> tuple:
    """
    Valide qu'un fichier audio existe et est au bon format
    
    Args:
        audio_path: Chemin vers le fichier audio
        
    Returns:
        Tuple (is_valid: bool, message: str)
    """
    audio_path = Path(audio_path)
    
    # Vérifier l'existence
    if not audio_path.exists():
        return False, f"Fichier non trouvé: {audio_path}"
    
    # Vérifier le format
    supported_formats = {'.wav', '.mp3', '.m4a', '.flac'}
    if audio_path.suffix.lower() not in supported_formats:
        return False, f"Format non supporté: {audio_path.suffix}"
    
    # Vérifier la taille (max 50 MB)
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 50:
        return False, f"Fichier trop volumineux: {file_size_mb:.2f}MB > 50MB"
    
    return True, "Fichier valide"


def get_audio_duration(audio_path: str) -> float:
    """
    Retourne la durée du fichier audio en secondes
    
    Args:
        audio_path: Chemin vers le fichier audio
        
    Returns:
        Durée en secondes (float)
    """
    try:
        is_valid, msg = validate_audio_file(audio_path)
        if not is_valid:
            raise ValueError(msg)
        
        # Charger les métadonnées sans charger l'audio complet
        duration = librosa.get_duration(filename=audio_path)
        return duration
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la durée: {e}")
        raise


def check_audio_silent(audio_array: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Vérifie si l'audio est complètement silencieux
    
    Args:
        audio_array: Array audio numpy
        threshold: Seuil RMS pour déterminer le silence
        
    Returns:
        True si silencieux, False sinon
    """
    if audio_array is None or len(audio_array) == 0:
        return True
    
    # Calculer l'énergie (RMS)
    rms = np.sqrt(np.mean(audio_array ** 2))
    
    is_silent = rms < threshold
    
    if is_silent:
        logger.warning(f"Audio silencieux détecté (RMS={rms:.6f})")
    
    return is_silent


def get_file_size_mb(file_path: str) -> float:
    """Retourne la taille du fichier en MB"""
    return Path(file_path).stat().st_size / (1024 * 1024)


def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes au format MM:SS
    
    Args:
        seconds: Durée en secondes
        
    Returns:
        Durée formatée (ex: "01:23")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def print_separator(title: str = "", width: int = 60):
    """Affiche un séparateur avec un titre optionnel"""
    if title:
        print(f"\n{'='*width}")
        print(f"  {title}")
        print(f"{'='*width}\n")
    else:
        print(f"\n{'-'*width}\n")
