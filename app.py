"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Gradio Application Module
=========================
This module provides an alternative web interface for NovaVision using Gradio.
Gradio offers a simple, component-based UI framework that's particularly well-suited
for ML demos and allows easy sharing via temporary public URLs.

Application Features:
    - Text input for emotional content or scene descriptions
    - Style selection dropdown (Artistic, Nature, Abstract, Dreamscape, Photorealistic)
    - Real-time image generation with progress tracking
    - Expandable analysis section showing emotion breakdown and enhanced prompt
    - Custom CSS styling matching the main index.html design

UI Components:
    - Text input area with placeholder text
    - Style dropdown selector
    - Generate button with gradient styling
    - Output image display
    - Collapsible emotion analysis panel
    - Enhanced prompt viewer

Usage:
    python app.py
    # Server starts at http://localhost:7860
"""

# Standard library imports
# (none required for this module)

# Third-party imports
import gradio as gr  # Gradio web UI framework for ML applications
from dotenv import load_dotenv  # Environment variable loading from .env files

# Local imports
from src.services.emotion_analyzer import EmotionAnalyzer, EmotionResult  # Emotion analysis
from src.services.image_generator import ImageGenerator  # Image generation service

# Load environment variables from .env file
load_dotenv()

# Global service instances (lazy-loaded)
# Using module-level variables with getter functions implements lazy initialization
_analyzer = None
_generator = None


def get_analyzer():
    """
    Get or create the emotion analyzer instance (lazy loading).

    Implements lazy initialization to defer model loading until first use,
    reducing application startup time.

    Returns:
        EmotionAnalyzer: Singleton instance of the emotion analyzer.
    """
    global _analyzer
    if _analyzer is None:
        print("[App] Initializing EmotionAnalyzer...")
        _analyzer = EmotionAnalyzer()
    return _analyzer


def get_generator():
    """
    Get or create the image generator instance (lazy loading).

    Implements lazy initialization to defer API client creation until
    first generation request.

    Returns:
        ImageGenerator: Singleton instance of the image generator.

    Requires:
        HF_TOKEN environment variable to be set.
    """
    global _generator
    if _generator is None:
        print("[App] Initializing ImageGenerator...")
        _generator = ImageGenerator()
    return _generator


def format_emotion_html(emotion_result: EmotionResult) -> str:
    """
    Format emotion analysis results as styled HTML for display.

    Creates a visually appealing HTML panel showing:
    - Primary emotion with confidence percentage
    - Valence and arousal metrics in separate cards
    - Bar chart visualization of all emotion scores

    Args:
        emotion_result: EmotionResult object from emotion analysis.

    Returns:
        HTML string containing the formatted emotion display panel.

    Example:
        >>> html = format_emotion_html(emotion_result)
        >>> gr.HTML(html)  # Display in Gradio
    """
    # Sort emotions by score (highest first)
    sorted_emotions = sorted(
        emotion_result.all_emotions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Build HTML structure with inline styles
    html = f"""
    <div style="padding: 16px; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;">
        <div style="margin-bottom: 16px;">
            <span style="font-size: 24px; font-weight: 600; color: #6366f1; text-transform: capitalize;">
                {emotion_result.primary_emotion}
            </span>
            <span style="font-size: 14px; color: #64748b; margin-left: 8px;">
                {emotion_result.confidence:.1%} confidence
            </span>
        </div>
        <div style="display: flex; gap: 16px; margin-bottom: 16px;">
            <div style="padding: 8px 12px; background: #fff; border-radius: 8px; border: 1px solid #e2e8f0;">
                <span style="font-size: 11px; color: #64748b; text-transform: uppercase;">Valence</span>
                <div style="font-size: 16px; font-weight: 500; color: {'#22c55e' if emotion_result.valence > 0 else '#ef4444' if emotion_result.valence < 0 else '#64748b'};">
                    {emotion_result.valence:+.1f}
                </div>
            </div>
            <div style="padding: 8px 12px; background: #fff; border-radius: 8px; border: 1px solid #e2e8f0;">
                <span style="font-size: 11px; color: #64748b; text-transform: uppercase;">Arousal</span>
                <div style="font-size: 16px; font-weight: 500; color: #0f172a;">
                    {emotion_result.arousal:.1f}
                </div>
            </div>
        </div>
        <div style="font-size: 12px; color: #64748b; margin-bottom: 8px; text-transform: uppercase; font-weight: 600;">
            All Emotions
        </div>
    """

    # Add bar chart for each emotion
    for emotion, score in sorted_emotions:
        # Use gradient color for primary emotion, gray for others
        bar_color = "linear-gradient(90deg, #6366f1, #8b5cf6)" if emotion == emotion_result.primary_emotion else "#cbd5e1"
        html += f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="width: 70px; font-size: 13px; color: #64748b; text-transform: capitalize;">{emotion}</span>
            <div style="flex: 1; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden;">
                <div style="width: {score * 100}%; height: 100%; background: {bar_color}; border-radius: 3px;"></div>
            </div>
            <span style="width: 45px; font-size: 12px; color: #94a3b8; text-align: right;">{score:.1%}</span>
        </div>
        """

    html += "</div>"
    return html


def generate_visualization(text: str, style: str, progress=gr.Progress()):
    """
    Main generation function for the Gradio interface.

    This function orchestrates the complete emotion-to-image pipeline:
    1. Validates user input
    2. Analyzes emotional content
    3. Generates image based on emotion and style
    4. Formats results for display

    Args:
        text: User input text describing emotions or scene.
        style: Visual style selection from dropdown.
        progress: Gradio progress tracker for status updates.

    Returns:
        Tuple of (image, emotion_html, prompt_text):
        - image: PIL Image object or None on error
        - emotion_html: Formatted HTML string for emotion display
        - prompt_text: The enhanced prompt used for generation

    Example:
        >>> image, html, prompt = generate_visualization(
        ...     "I feel peaceful",
        ...     "Nature"
        ... )
    """
    # Validate input
    if not text or not text.strip():
        return None, "<p style='color: #ef4444;'>Please enter some text to analyze.</p>", ""

    try:
        # Step 1: Analyze emotion with progress tracking
        progress(0.1, desc="Analyzing emotions...")
        print(f"[App] Processing: '{text[:50]}...'")

        analyzer = get_analyzer()
        emotion_result = analyzer.analyze(text)

        progress(0.3, desc=f"Detected: {emotion_result.primary_emotion}")

        # Step 2: Generate image
        progress(0.4, desc="Generating visualization...")

        generator = get_generator()
        result = generator.generate(
            text=text,
            emotion_result=emotion_result,
            style=style.lower(),
            width=1024,
            height=1024
        )

        progress(0.9, desc="Finalizing...")

        # Step 3: Format outputs for display
        emotion_html = format_emotion_html(emotion_result)
        prompt_text = result.prompt

        progress(1.0, desc="Complete!")

        return result.image, emotion_html, prompt_text

    except Exception as e:
        # Log error and return error message
        print(f"[App] Error: {e}")
        import traceback
        traceback.print_exc()
        error_html = f"<p style='color: #ef4444;'>Error: {str(e)}</p>"
        return None, error_html, ""


# Custom CSS Styling
# ==================
# This CSS matches the design of the main index.html for visual consistency.
# It customizes Gradio's default styling with the NovaVision color scheme.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: linear-gradient(180deg, #fafafa 0%, #f5f5f7 100%) !important;
}

.app-header {
    text-align: center;
    padding: 40px 0 32px;
}

.app-title {
    font-size: 48px !important;
    font-weight: 800 !important;
    letter-spacing: -2px !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px !important;
}

.app-subtitle {
    font-size: 18px !important;
    color: #64748b !important;
    font-weight: 400 !important;
}

.input-card, .output-card {
    background: #fafafa !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    transition: box-shadow 0.3s ease !important;
}

.input-card:hover, .output-card:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06) !important;
}

textarea {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.12) !important;
}

.generate-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    background-size: 200% 200% !important;
    background-position: 0% 50% !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s ease !important;
}

.generate-btn:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.45) !important;
    background-position: 100% 50% !important;
}

.generate-btn:active {
    transform: translateY(0) scale(0.98) !important;
}

.output-image {
    border-radius: 12px !important;
    overflow: hidden !important;
}

footer {
    display: none !important;
}

::selection {
    background: rgba(99, 102, 241, 0.2);
}
"""


# Build the Gradio Interface
# ==========================
# Using gr.Blocks for full layout control and custom styling
with gr.Blocks(title="NovaVision") as app:

    # Header Section
    gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">NovaVision</h1>
            <p class="app-subtitle">Transform emotions into AI-generated art</p>
        </div>
    """)

    # Main Content Row
    with gr.Row():
        # Input Column (narrower)
        with gr.Column(scale=2):
            with gr.Group(elem_classes="input-card"):
                # Text input area
                text_input = gr.Textbox(
                    label="How are you feeling today?",
                    placeholder="Share a thought, memory, or describe how you're feeling...",
                    lines=5,
                    max_lines=8
                )

                # Style selection dropdown
                style_dropdown = gr.Dropdown(
                    choices=["Artistic", "Nature", "Abstract", "Dreamscape", "Photorealistic"],
                    value="Artistic",
                    label="Visual Style"
                )

                # Generate button
                generate_btn = gr.Button(
                    "Generate Visualization",
                    variant="primary",
                    elem_classes="generate-btn"
                )

        # Output Column (wider)
        with gr.Column(scale=3):
            with gr.Group(elem_classes="output-card"):
                # Generated image display
                output_image = gr.Image(
                    label="Generated Visualization",
                    show_label=False,
                    type="pil",
                    height=450,
                    elem_classes="output-image"
                )

    # Analysis Section (collapsible)
    with gr.Accordion("View Analysis", open=False):
        with gr.Row():
            with gr.Column():
                # Emotion analysis HTML display
                emotion_output = gr.HTML(label="Emotion Analysis")
            with gr.Column():
                # Enhanced prompt display
                prompt_output = gr.Textbox(
                    label="Enhanced Prompt",
                    lines=4,
                    interactive=False
                )

    # Footer Section
    gr.HTML("""
        <div style="text-align: center; padding: 32px 0; color: #94a3b8; font-size: 13px;">
            <div style="font-weight: 600; color: #64748b; margin-bottom: 4px;">NovaVision</div>
            <div>AI-powered emotion visualization</div>
            <div style="margin-top: 8px;">2025 NovaVision. All rights reserved.</div>
        </div>
    """)

    # Connect generate button to handler function
    generate_btn.click(
        fn=generate_visualization,
        inputs=[text_input, style_dropdown],
        outputs=[output_image, emotion_output, prompt_output]
    )


# Main entry point
if __name__ == "__main__":
    print("=" * 50)
    print("Starting NovaVision")
    print("=" * 50)

    # Pre-load models to reduce first-request latency
    print("\n[App] Pre-loading models...")
    get_analyzer()
    get_generator()
    print("[App] Models loaded!")

    # Launch the Gradio interface
    print("\n[App] Launching Gradio interface...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS
    )
