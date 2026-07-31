"""
Interface Gradio - Analyse de Sentiment Vocal
Design minimaliste et professionnel
"""

import gradio as gr
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import SentimentAnalysisPipeline
from src.utils import format_duration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Initialisation du pipeline...")
    pipeline = SentimentAnalysisPipeline(device="cpu")
    logger.info("✓ Pipeline chargé")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation: {e}")
    pipeline = None


def process_audio(audio_file):
    """Traite un fichier audio et retourne transcription + sentiment"""
    
    if pipeline is None:
        return (
            "Erreur: Pipeline non initialisé",
            {"ERREUR": 0.0},
            0.0,
            "Le pipeline n'a pas pu être chargé"
        )
    
    if not audio_file:
        return (
            "Veuillez télécharger un fichier audio",
            {"NEUTRE": 0.0},
            0.0,
            "Aucun fichier sélectionné"
        )
    
    try:
        if isinstance(audio_file, tuple):
            audio_path = audio_file[0]
        else:
            audio_path = audio_file
        
        logger.info(f"Traitement du fichier: {audio_path}")
        result = pipeline.process(audio_path)
        
        transcription = result['transcription']
        sentiment = result['sentiment']
        confidence = result['confidence']
        duration = result['duration']
        
        # Mapper les sentiments français
        sentiment_map = {
            'POSITIVE': 'Positif',
            'NEGATIVE': 'Négatif',
            'NEUTRAL': 'Neutre'
        }
        sentiment_display = sentiment_map.get(sentiment, sentiment)
        sentiment_dict = {sentiment_display: float(confidence)}
        
        # Message simple et épuré
        message = f"Analyse réussie | Durée: {format_duration(duration)} | Confiance: {confidence:.1%}"
        
        logger.info(f"✓ Résultat: {sentiment} ({confidence:.2%})")
        return transcription, sentiment_dict, confidence, message
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        return (
            f"Erreur: {str(e)}",
            {"ERREUR": 0.0},
            0.0,
            f"Erreur: {str(e)}"
        )


def create_interface():
    """Crée l'interface Gradio minimaliste"""
    
    title = "Analyse de Sentiment Vocal"
    
    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(),
        css="""
        .header {
            padding: 30px 0;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 30px;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2em;
            color: #333;
            font-weight: 600;
        }
        
        .header p {
            margin: 5px 0 0 0;
            color: #666;
            font-size: 1em;
        }
        
        .section-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #333;
            margin-top: 20px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ddd;
        }
        
        .info-box {
            background: #f5f5f5;
            padding: 15px;
            border-left: 3px solid #666;
            border-radius: 4px;
            margin: 15px 0;
            font-size: 0.95em;
            color: #555;
        }
        
        .info-box strong {
            color: #333;
        }
        
        .info-box ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .info-box li {
            margin: 5px 0;
        }
        
        .result-text {
            padding: 12px;
            background: #fafafa;
            border-radius: 4px;
            color: #555;
            font-size: 0.95em;
        }
        
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class='header'>
            <h1>Analyse de Sentiment Vocal</h1>
            <p>Transcription et analyse automatique d'appels vocaux en français</p>
        </div>
        """)
        
        # Description simple
        gr.HTML("""
        <div class='info-box'>
            <strong>Fonctionnalités:</strong>
            <ul>
                <li><strong>Transcription:</strong> Conversion audio → texte (Wav2Vec 2.0)</li>
                <li><strong>Sentiment:</strong> Classification Positif / Négatif / Neutre (BERT)</li>
                <li><strong>Confiance:</strong> Score de précision 0-100%</li>
            </ul>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div class='section-title'>Entrée Audio</div>")
                
                audio_input = gr.Audio(
                    label="Télécharger ou enregistrer un fichier audio",
                    type="filepath",
                    interactive=True
                )
                
                submit_btn = gr.Button(
                    "Analyser",
                    variant="primary",
                    size="lg",
                    scale=1
                )
                
                gr.HTML("""
                <div class='info-box'>
                    <strong>Guide d'utilisation</strong>
                    <ul>
                        <li>Téléchargez un fichier audio (WAV, MP3, FLAC, M4A)</li>
                        <li>Ou enregistrez directement avec votre microphone</li>
                        <li>Cliquez sur "Analyser" pour traiter l'audio</li>
                    </ul>
                    <p style='margin: 10px 0 0 0;'><strong>Durée maximale:</strong> 5 minutes</p>
                </div>
                """)
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='section-title'>Transcription</div>")
                transcription_output = gr.Textbox(
                    label="Texte transcrit",
                    lines=6,
                    interactive=False,
                    placeholder="Le texte transcrit apparaîtra ici...",
                    show_copy_button=True
                )
            
            with gr.Column():
                gr.HTML("<div class='section-title'>Résultats</div>")
                
                sentiment_output = gr.Label(
                    label="Sentiment détecté",
                    show_label=False
                )
                
                confidence_output = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Niveau de confiance",
                    interactive=False
                )
                
                gr.HTML("<div class='section-title' style='border-bottom: 2px solid #ddd;'>Détails</div>")
                message_output = gr.Textbox(
                    label="",
                    lines=2,
                    interactive=False,
                    placeholder="Les détails s'afficheront ici...",
                    elem_classes="result-text"
                )
        
        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[
                transcription_output,
                sentiment_output,
                confidence_output,
                message_output
            ],
            show_progress="minimal"
        )
        
        # Footer
        gr.HTML("""
        <div class='footer'>
            <p>Projet DIT - Deep Learning 2 (2026)</p>
            <p>Modèles: Wav2Vec 2.0 (français) + BERT Multilingual</p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
