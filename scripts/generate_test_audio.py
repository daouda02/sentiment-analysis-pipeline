"""
Script pour générer les fichiers audio de test via TTS (Text-to-Speech)
Génère 3 fichiers : positif, négatif, neutre
"""

from gtts import gTTS
from pathlib import Path
import csv


# Textes pour chaque sentiment
TEST_TEXTS = {
    "positive": "Je suis très satisfait de votre service. C'est excellent! J'ai apprécié votre professionnalisme et votre aide. Je recommande vivement.",
    "negative": "Je ne suis pas du tout content. Le service était horrible. J'ai attendu très longtemps et rien n'a été résolu. C'est inacceptable!",
    "neutral": "Bonjour, je vous appelle pour connaître les horaires d'ouverture. Pouvez-vous me dire quand vous fermez? Je voudrais avoir cette information."
}


def generate_test_audio():
    """Génère les fichiers de test via TTS"""
    
    output_dir = Path("data/test_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Génération des fichiers de test via TTS...")
    print("=" * 60)
    
    annotations = []
    
    for sentiment, text in TEST_TEXTS.items():
        output_file = output_dir / f"{sentiment}.wav"
        
        print(f"\n Génération : {sentiment.upper()}")
        print(f"   Texte: {text[:50]}...")
        print(f"   Fichier: {output_file}")
        
        try:
            # Générer le fichier audio via gTTS
            tts = gTTS(text=text, lang='fr', slow=False)
            tts.save(str(output_file))
            
            print(f"Généré avec succès")
            
            # Ajouter à la liste d'annotations
            annotations.append({
                'filename': output_file.name,
                'sentiment': sentiment.upper(),
                'text': text
            })
        
        except Exception as e:
            print(f"Erreur: {e}")
    
    # Sauvegarder les annotations
    annotations_file = output_dir.parent / "annotations.csv"
    
    print(f"\n Sauvegarde des annotations: {annotations_file}")
    
    with open(annotations_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'sentiment', 'text'])
        writer.writeheader()
        writer.writerows(annotations)
    
    print(f" {len(annotations)} fichiers générés")
    print("=" * 60)


if __name__ == "__main__":
    generate_test_audio()
