"""
Script de démonstration du pipeline
Utilise les fichiers de test dans data/test_audio/

Exécution : python main.py
"""
import os
import sys
from pathlib import Path
import logging

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
# Ajouter le dossier src au chemin
sys.path.insert(0, str(Path(__file__).parent))

    
from src.pipeline import SentimentAnalysisPipeline
from src.utils import print_separator, format_duration

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo():
    """Lance une démo du pipeline sur les fichiers de test"""
    
    print_separator("SENTIMENT ANALYSIS PIPELINE - DÉMO")
    
    # Initialiser le pipeline
    try:
        logger.info("Initialisation du pipeline...")
        pipeline = SentimentAnalysisPipeline(device="cpu")
        logger.info("Pipeline chargé avec succès\n")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        return
    
    # Chercher les fichiers de test
    test_audio_dir = Path("data/test_audio")
    
    if not test_audio_dir.exists():
        print(f" Dossier non trouvé: {test_audio_dir}")
        print(f"\n Créez d'abord les fichiers de test:")
        print(f"   python scripts/generate_test_audio.py")
        return
    
    audio_files = sorted(test_audio_dir.glob("*.wav"))
    
    if not audio_files:
        print(f" Pas de fichiers .wav trouvés dans {test_audio_dir}")
        print(f"\n Génération des fichiers de test:")
        print(f"   python scripts/generate_test_audio.py")
        return
    
    print(f"Fichiers trouvés: {len(audio_files)}\n")
    
    # Traiter chaque fichier
    results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*60}")
        print(f" Fichier {i}/{len(audio_files)}: {audio_file.name}")
        print(f"{'='*60}")
        
        try:
            # Traiter le fichier
            result = pipeline.process(str(audio_file))
            
            # Afficher les résultats
            print(f"\n TRANSCRIPTION:")
            print(f"   {result['transcription']}")
            
            print(f"\n SENTIMENT:")
            print(f"   Classification: {result['sentiment']}")
            print(f"   Confiance: {result['confidence']:.1%}")
            print(f"   Score: {result['score']:.4f}")
            
            print(f"\n DÉTAILS:")
            print(f"   Durée audio: {format_duration(result['duration'])}")
            print(f"   Modèles utilisés:")
            print(f"     - ASR: {result['models_used']['asr']['model_name']}")
            print(f"     - Sentiment: {result['models_used']['sentiment']['model_name']}")
            
            results.append({
                'file': audio_file.name,
                'sentiment': result['sentiment'],
                'confidence': result['confidence']
            })
            
            print(f"\n Traitement réussi")
        
        except Exception as e:
            logger.error(f" Erreur lors du traitement: {e}")
            results.append({
                'file': audio_file.name,
                'error': str(e)
            })
    
    # Résumé final
    print_separator("RÉSUMÉ DES RÉSULTATS")
    
    print(f"{'Fichier':<20} | {'Sentiment':<10} | {'Confiance':<10}")
    print("-" * 45)
    
    for result in results:
        if 'error' in result:
            print(f"{result['file']:<20} | {'ERROR':<10} | {'-':<10}")
        else:
            confidence = f"{result['confidence']:.0%}"
            print(f"{result['file']:<20} | {result['sentiment']:<10} | {confidence:<10}")
    
    print(f"\n Démo complétée ({len(results)} fichiers traités)")
    print("="*60 + "\n")


if __name__ == "__main__":
    demo()
