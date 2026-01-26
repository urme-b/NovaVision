"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Flask Server Module
===================
This module implements the Flask-based web server for NovaVision. It serves the
main index.html single-page application and provides RESTful API endpoints for
emotion analysis and image generation.

Server Architecture:
    - Serves static files (index.html) from the root directory
    - Provides two main API endpoints: /api/analyze and /api/generate
    - Uses lazy loading for ML models to optimize startup time
    - Enables CORS for cross-origin requests (development support)

API Endpoints:
    GET  /              - Serve the main web interface
    POST /api/analyze   - Real-time emotion analysis (lightweight)
    POST /api/generate  - Full emotion-to-image generation pipeline

Usage:
    python server.py
    # Server starts at http://localhost:8000
"""

# Standard library imports
import os  # Environment variable access
import base64  # Base64 encoding for image data
from io import BytesIO  # In-memory binary streams

# Third-party imports
from flask import Flask, request, jsonify, send_from_directory  # Flask web framework
from flask_cors import CORS  # Cross-Origin Resource Sharing support
from dotenv import load_dotenv  # Environment variable loading from .env files

# Local imports
from src.services.emotion_analyzer import EmotionAnalyzer  # Emotion analysis service
from src.services.image_generator import ImageGenerator  # Image generation service

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
# static_folder='.' allows serving index.html from the project root
app = Flask(__name__, static_folder='.')

# Enable CORS for all routes (allows frontend development with different origins)
CORS(app)

# Global service instances (lazy-loaded)
# Using module-level variables with getter functions implements a simple
# singleton pattern that defers model loading until first request
_analyzer = None
_generator = None


def get_analyzer() -> EmotionAnalyzer:
    """
    Get or create the emotion analyzer instance (lazy loading).

    This function implements lazy initialization of the EmotionAnalyzer.
    The model is only loaded when first needed, reducing startup time
    and allowing the server to start quickly.

    Returns:
        EmotionAnalyzer: Singleton instance of the emotion analyzer.

    Note:
        Thread-safe in most WSGI contexts due to GIL, but consider
        using threading.Lock for production multi-threaded deployments.
    """
    global _analyzer
    if _analyzer is None:
        print("[Server] Initializing EmotionAnalyzer...")
        _analyzer = EmotionAnalyzer()
    return _analyzer


def get_generator() -> ImageGenerator:
    """
    Get or create the image generator instance (lazy loading).

    This function implements lazy initialization of the ImageGenerator.
    The client is only created when first needed, deferring API
    authentication until actually required.

    Returns:
        ImageGenerator: Singleton instance of the image generator.

    Note:
        Requires HF_TOKEN environment variable to be set.
    """
    global _generator
    if _generator is None:
        print("[Server] Initializing ImageGenerator...")
        _generator = ImageGenerator()
    return _generator


@app.route('/')
def serve_index() -> str:
    """
    Serve the main index.html single-page application.

    This route handler serves the frontend web interface. The index.html
    file contains the complete SPA with embedded CSS and JavaScript.

    Returns:
        Response: The index.html file content.
    """
    return send_from_directory('.', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze() -> tuple:
    """
    Lightweight emotion analysis endpoint for real-time updates.

    This endpoint provides fast emotion analysis without image generation,
    suitable for real-time feedback as users type. The frontend uses this
    for live emotion bar chart updates with debouncing.

    Request Body (JSON):
        text (str): The text to analyze for emotional content.

    Returns:
        JSON response with:
        - success (bool): Whether analysis succeeded
        - emotions (list): Sorted list of {name, score} for all emotions
        - primary_emotion (str): The dominant detected emotion
        - confidence (float): Confidence percentage (0-100)
        - valence (float): Emotional valence (-1 to +1)
        - arousal (float): Emotional arousal (0 to 1)

    Error Responses:
        400: Text too short (less than 3 characters)
        500: Internal server error during analysis

    Example:
        POST /api/analyze
        {"text": "I feel happy today!"}

        Response:
        {
            "success": true,
            "emotions": [{"name": "joy", "score": 87.5}, ...],
            "primary_emotion": "joy",
            "confidence": 87.5,
            "valence": 0.8,
            "arousal": 0.7
        }
    """
    try:
        # Parse JSON request body
        data = request.get_json()
        text = data.get('text', '').strip()

        # Validate input length (minimum 3 characters for meaningful analysis)
        if not text or len(text) < 3:
            return jsonify({'error': 'Text too short'}), 400

        # Get analyzer and perform emotion detection
        analyzer = get_analyzer()
        emotion_result = analyzer.analyze(text)

        # Sort emotions by score (highest first) for display
        sorted_emotions = sorted(
            emotion_result.all_emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return formatted response
        return jsonify({
            'success': True,
            'emotions': [
                {'name': e[0], 'score': round(e[1] * 100, 1)}
                for e in sorted_emotions
            ],
            'primary_emotion': emotion_result.primary_emotion,
            'confidence': round(emotion_result.confidence * 100, 1),
            'valence': emotion_result.valence,
            'arousal': emotion_result.arousal
        })

    except Exception as e:
        # Return error message on failure
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate() -> tuple:
    """
    Full emotion-to-image generation endpoint.

    This endpoint runs the complete NovaVision pipeline:
    1. Analyze text for emotional content
    2. Build optimized prompt based on emotion and style
    3. Generate image using FLUX.1 model
    4. Return image as base64-encoded data URL

    Request Body (JSON):
        text (str): The text to analyze and visualize
        style (str): Visual style preset (artistic, nature, abstract, etc.)
        seed (int, optional): Random seed for reproducible generation

    Returns:
        JSON response with:
        - success (bool): Whether generation succeeded
        - image (str): Base64-encoded PNG data URL
        - emotions (list): Sorted emotion scores
        - primary_emotion (str): Detected emotion
        - confidence (float): Emotion confidence
        - valence (float): Emotional valence
        - arousal (float): Emotional arousal
        - prompt (str): The enhanced prompt used for generation
        - original_text (str): The original user input
        - style (str): The style preset used
        - input_type (str): 'emotion' or 'object'
        - seed (int): The seed used for generation
        - timestamp (str): ISO format timestamp

    Error Responses:
        400: Empty text provided
        500: Internal server error during generation

    Example:
        POST /api/generate
        {"text": "I feel peaceful", "style": "nature", "seed": 42}
    """
    try:
        # Parse JSON request body
        data = request.get_json()
        text = data.get('text', '').strip()
        style = data.get('style', 'artistic')
        seed = data.get('seed', None)  # Optional seed for reproducibility

        # Validate input
        if not text:
            return jsonify({'error': 'Please enter some text to analyze.'}), 400

        print(f"[Server] Processing: '{text[:50]}...'")
        print(f"[Server] Style: {style}, Seed: {seed}")

        # Step 1: Analyze emotion in the input text
        analyzer = get_analyzer()
        emotion_result = analyzer.analyze(text)

        # Step 2: Generate image using emotion context
        generator = get_generator()
        result = generator.generate(
            text=text,
            emotion_result=emotion_result,
            style=style.lower(),
            width=1024,
            height=1024,
            seed=seed
        )

        # Step 3: Convert PIL image to base64 data URL
        # This allows embedding the image directly in JSON response
        buffered = BytesIO()
        result.image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Sort emotions by score for display
        sorted_emotions = sorted(
            emotion_result.all_emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return comprehensive response with all metadata
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'emotions': [
                {'name': e[0], 'score': round(e[1] * 100, 1)}
                for e in sorted_emotions
            ],
            'primary_emotion': emotion_result.primary_emotion,
            'confidence': round(emotion_result.confidence * 100, 1),
            'valence': emotion_result.valence,
            'arousal': emotion_result.arousal,
            'prompt': result.prompt,
            'original_text': text,
            'style': style,
            'input_type': result.input_type,
            'seed': result.seed,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })

    except Exception as e:
        # Log error details for debugging
        print(f"[Server] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Main entry point
if __name__ == '__main__':
    print("=" * 50)
    print("Starting NovaVision Server")
    print("=" * 50)

    # Pre-load models to reduce first-request latency
    print("\n[Server] Pre-loading models...")
    get_analyzer()
    get_generator()
    print("[Server] Models loaded!")

    # Start the Flask development server
    print("\n[Server] Starting Flask server...")
    print("[Server] Open http://localhost:8000 in your browser")
    app.run(host='0.0.0.0', port=8000, debug=False)
