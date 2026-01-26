"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Emotion Analyzer Tests
======================
This module contains unit tests for the EmotionAnalyzer service.
Tests verify correct emotion detection across different emotional
categories and validate the structure of analysis results.

Test Categories:
    - Emotion type validation
    - Joy detection accuracy
    - Sadness detection accuracy
    - Result structure completeness

Running Tests:
    pytest tests/test_emotion_analyzer.py -v
"""

# Third-party imports
import pytest  # Testing framework

# Local imports
from src.models.schemas import EmotionType  # Emotion enumeration


class TestEmotionAnalyzer:
    """
    Test suite for the EmotionAnalyzer service.

    This class contains test cases that verify the emotion analyzer
    correctly identifies emotional content in text and returns
    properly structured results.

    Test Methods:
        - test_emotion_types_exist: Verify all emotion types are defined
        - test_joy_detection: Test positive emotion detection
        - test_sadness_detection: Test negative emotion detection
        - test_all_scores_present: Verify complete score output
    """

    def test_emotion_types_exist(self):
        """
        Verify all expected emotion types are defined in the enum.

        This test ensures that all seven emotion categories supported
        by the model are properly defined in the EmotionType enum.
        """
        # List of expected emotion categories
        expected_emotions = [
            "anger", "disgust", "fear", "joy",
            "neutral", "sadness", "surprise"
        ]

        # Verify each emotion can be instantiated from the enum
        for emotion in expected_emotions:
            assert EmotionType(emotion) is not None

    def test_joy_detection(self):
        """
        Test that joyful/happy text is correctly identified.

        This test verifies that clearly positive, happy text is
        classified as 'joy' with high confidence.
        """
        from src.services.emotion_analyzer import get_emotion_analyzer

        analyzer = get_emotion_analyzer()
        result = analyzer.analyze("I am so happy and excited today!")

        # Verify joy is detected as the primary emotion
        assert result.emotion == EmotionType.JOY

        # Verify confidence is reasonably high
        assert result.confidence > 0.5

    def test_sadness_detection(self):
        """
        Test that sad/melancholic text is correctly identified.

        This test verifies that clearly negative, sad text is
        classified as 'sadness' with high confidence.
        """
        from src.services.emotion_analyzer import get_emotion_analyzer

        analyzer = get_emotion_analyzer()
        result = analyzer.analyze("I feel so lonely and heartbroken.")

        # Verify sadness is detected as the primary emotion
        assert result.emotion == EmotionType.SADNESS

        # Verify confidence is reasonably high
        assert result.confidence > 0.5

    def test_all_scores_present(self):
        """
        Verify all seven emotion scores are returned in results.

        This test ensures the analyzer returns scores for all
        emotion categories, with values properly bounded between 0 and 1.
        """
        from src.services.emotion_analyzer import get_emotion_analyzer

        analyzer = get_emotion_analyzer()
        result = analyzer.analyze("This is a test.")

        # Verify all 7 emotions have scores
        assert len(result.all_scores) == 7

        # Verify all scores are within valid range [0, 1]
        assert all(0 <= score <= 1 for score in result.all_scores.values())
