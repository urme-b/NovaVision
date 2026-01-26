"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Emotion Analyzer Service
========================
This module provides emotion detection and analysis capabilities using the HuggingFace
Transformers library. It analyzes text input to detect emotional content across seven
emotion categories and provides both discrete emotion labels and dimensional affect
measurements (valence and arousal).

Model Used:
    j-hartmann/emotion-english-distilroberta-base
    - A fine-tuned DistilRoBERTa model for emotion classification
    - Trained on multiple emotion datasets (GoEmotions, ISEAR, etc.)
    - Supports 7 emotion categories with multi-label classification

Output Format:
    EmotionResult dataclass containing:
    - primary_emotion: The dominant detected emotion (str)
    - confidence: Confidence score for primary emotion (0.0-1.0)
    - valence: Emotional valence from -1 (negative) to +1 (positive)
    - arousal: Emotional intensity from 0 (calm) to 1 (energized)
    - all_emotions: Dictionary of all emotion scores

Emotion Categories:
    - joy: Happiness, pleasure, positive feelings
    - sadness: Grief, sorrow, melancholy
    - anger: Frustration, irritation, rage
    - fear: Anxiety, worry, dread
    - surprise: Astonishment, amazement, shock
    - disgust: Revulsion, distaste, aversion
    - neutral: Absence of strong emotion
"""

# Standard library imports
from dataclasses import dataclass

# Third-party imports
from transformers import pipeline  # HuggingFace transformers for NLP model inference


@dataclass
class EmotionResult:
    """
    Container for emotion analysis results.

    This dataclass holds the complete output from emotion analysis, including
    the primary detected emotion, confidence metrics, dimensional affect
    measurements, and scores for all emotion categories.

    Attributes:
        primary_emotion: The emotion with the highest confidence score.
                        One of: joy, sadness, anger, fear, surprise, disgust, neutral
        confidence: Confidence score for the primary emotion (range: 0.0 to 1.0).
                   Higher values indicate stronger model certainty.
        valence: Emotional valence indicating positivity/negativity.
                Range: -1.0 (very negative) to +1.0 (very positive).
                Derived from emotion-to-valence mapping.
        arousal: Emotional arousal indicating intensity/energy level.
                Range: 0.0 (calm/low energy) to 1.0 (excited/high energy).
                Derived from emotion-to-arousal mapping.
        all_emotions: Dictionary mapping each emotion category to its
                     confidence score. All scores sum to approximately 1.0.

    Example:
        >>> result = EmotionResult(
        ...     primary_emotion="joy",
        ...     confidence=0.85,
        ...     valence=0.8,
        ...     arousal=0.7,
        ...     all_emotions={"joy": 0.85, "neutral": 0.10, ...}
        ... )
    """
    primary_emotion: str
    confidence: float
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # 0 (calm) to 1 (energized)
    all_emotions: dict[str, float]


# Valence and Arousal Dimensional Mappings
# =========================================
# These mappings convert discrete emotion categories to continuous dimensional
# values based on the circumplex model of affect (Russell, 1980).
#
# Valence: Indicates how positive or negative the emotion is
#   - Positive valence (>0): Pleasant emotions (joy, surprise)
#   - Negative valence (<0): Unpleasant emotions (sadness, anger, fear, disgust)
#   - Zero valence (=0): Neutral emotional state
#
# Arousal: Indicates the activation/energy level of the emotion
#   - High arousal (>0.5): Activating emotions (anger, fear, surprise, joy)
#   - Low arousal (<0.5): Deactivating emotions (sadness, neutral)
EMOTION_DIMENSIONS = {
    "joy": {"valence": 0.8, "arousal": 0.7},       # Positive, moderately activating
    "sadness": {"valence": -0.7, "arousal": 0.3},  # Negative, deactivating
    "anger": {"valence": -0.6, "arousal": 0.9},    # Negative, highly activating
    "fear": {"valence": -0.8, "arousal": 0.8},     # Very negative, highly activating
    "surprise": {"valence": 0.3, "arousal": 0.9},  # Slightly positive, highly activating
    "disgust": {"valence": -0.5, "arousal": 0.5},  # Negative, moderately activating
    "neutral": {"valence": 0.0, "arousal": 0.3},   # Zero valence, low arousal
}


class EmotionAnalyzer:
    """
    Emotion analysis service using HuggingFace Transformers.

    This class provides emotion detection capabilities by leveraging a pre-trained
    transformer model fine-tuned for emotion classification. It processes text input
    and returns detailed emotion analysis including confidence scores and dimensional
    affect measurements.

    The analyzer uses the j-hartmann/emotion-english-distilroberta-base model by default,
    which is optimized for English text and provides fast inference while maintaining
    good accuracy across the seven supported emotion categories.

    Attributes:
        classifier: HuggingFace pipeline object for text classification.

    Example:
        >>> analyzer = EmotionAnalyzer()
        >>> result = analyzer.analyze("I'm so happy today!")
        >>> print(result.primary_emotion)
        'joy'
        >>> print(result.confidence)
        0.92

    Note:
        Model loading happens during initialization and may take a few seconds
        on first run as weights are downloaded from HuggingFace Hub.
    """

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the emotion analyzer with the specified model.

        Args:
            model_name: HuggingFace model identifier for emotion classification.
                       Defaults to j-hartmann/emotion-english-distilroberta-base,
                       a DistilRoBERTa model fine-tuned on emotion datasets.

        Example:
            >>> analyzer = EmotionAnalyzer()  # Use default model
            >>> analyzer = EmotionAnalyzer("custom/emotion-model")  # Custom model
        """
        print(f"[EmotionAnalyzer] Loading model: {model_name}")

        # Initialize the HuggingFace text classification pipeline
        # top_k=None returns scores for all labels instead of just top-1
        # device=-1 forces CPU inference for compatibility
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,  # Return all emotion scores, not just the highest
            device=-1    # Use CPU (-1) for maximum compatibility; use 0 for GPU
        )
        print("[EmotionAnalyzer] Model loaded successfully")

    def analyze(self, text: str) -> EmotionResult:
        """
        Analyze text to detect emotional content.

        This method processes the input text through the emotion classification
        model and returns comprehensive analysis results including the primary
        emotion, confidence score, dimensional affect values, and all emotion
        category scores.

        Args:
            text: The input text to analyze for emotional content.
                 Should be non-empty and contain meaningful content.
                 Works best with English text of 1-512 tokens.

        Returns:
            EmotionResult containing:
            - primary_emotion: Detected emotion with highest confidence
            - confidence: Confidence score for primary emotion (0.0-1.0)
            - valence: Positivity/negativity score (-1.0 to +1.0)
            - arousal: Energy/intensity score (0.0 to 1.0)
            - all_emotions: Dict of all emotion category scores

        Raises:
            ValueError: If input text is empty or None.

        Example:
            >>> analyzer = EmotionAnalyzer()
            >>> result = analyzer.analyze("I feel so grateful and blessed today!")
            >>> print(f"Emotion: {result.primary_emotion}")
            Emotion: joy
            >>> print(f"Confidence: {result.confidence:.2%}")
            Confidence: 87.50%
            >>> print(f"Valence: {result.valence:+.1f}")
            Valence: +0.8
        """
        # Input validation - ensure we have meaningful text to analyze
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # Log the analysis request (truncate long inputs for readability)
        print(f"[EmotionAnalyzer] Analyzing text: '{text[:50]}...' " if len(text) > 50 else f"[EmotionAnalyzer] Analyzing text: '{text}'")

        # Run the text through the classification model
        # Returns list of dicts with 'label' and 'score' keys for each emotion
        results = self.classifier(text)[0]

        # Build dictionary of all emotion scores with lowercase keys for consistency
        all_emotions = {r["label"].lower(): r["score"] for r in results}
        print(f"[EmotionAnalyzer] Raw scores: {all_emotions}")

        # Find the primary emotion (highest confidence score)
        primary = max(results, key=lambda x: x["score"])
        primary_emotion = primary["label"].lower()
        confidence = primary["score"]

        print(f"[EmotionAnalyzer] Primary emotion: {primary_emotion} (confidence: {confidence:.3f})")

        # Map the discrete emotion to dimensional valence/arousal values
        # Fall back to neutral dimensions if emotion not found in mapping
        dimensions = EMOTION_DIMENSIONS.get(primary_emotion, {"valence": 0.0, "arousal": 0.5})
        valence = dimensions["valence"]
        arousal = dimensions["arousal"]

        print(f"[EmotionAnalyzer] Valence: {valence}, Arousal: {arousal}")

        # Construct and return the result object
        return EmotionResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            valence=valence,
            arousal=arousal,
            all_emotions=all_emotions
        )


# Module self-test - run when executed directly
if __name__ == "__main__":
    # Test the analyzer with sample inputs
    print("=" * 50)
    print("Testing EmotionAnalyzer")
    print("=" * 50)

    analyzer = EmotionAnalyzer()

    # Test cases covering different emotion categories
    test_texts = [
        "I feel happy and excited today",
        "This is so frustrating and annoying",
        "I'm scared about what might happen",
        "Just a normal day, nothing special",
    ]

    for text in test_texts:
        print(f"\n{'=' * 50}")
        result = analyzer.analyze(text)
        print(f"\nResult: {result}")
