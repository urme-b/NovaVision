"""
NovaVision - Unit Tests
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Service Unit Tests
==================
Unit tests for the core services: EmotionAnalyzer and ImageGenerator.
"""

import pytest
import sys
sys.path.insert(0, '.')

from src.services.emotion_analyzer import EmotionAnalyzer


class TestEmotionAnalyzer:
    """Test suite for EmotionAnalyzer service."""

    def setup_method(self):
        """Initialize analyzer before each test."""
        self.analyzer = EmotionAnalyzer()

    def test_happy_text_returns_joy(self):
        """Test that happy text is classified as joy."""
        result = self.analyzer.analyze("I feel so happy and excited today!")
        assert result.primary_emotion == "joy"
        assert result.confidence > 0.5

    def test_sad_text_returns_sadness(self):
        """Test that sad text is classified as sadness."""
        result = self.analyzer.analyze("I feel depressed and hopeless")
        assert result.primary_emotion == "sadness"
        assert result.confidence > 0.5

    def test_neutral_text_returns_neutral(self):
        """Test that neutral text returns neutral or low confidence."""
        result = self.analyzer.analyze("The table is made of wood")
        assert result.primary_emotion == "neutral" or result.confidence < 0.5

    def test_valence_positive_for_joy(self):
        """Test that joy has positive valence."""
        result = self.analyzer.analyze("I am thrilled and joyful!")
        assert result.valence > 0

    def test_valence_negative_for_sadness(self):
        """Test that sadness has negative valence."""
        result = self.analyzer.analyze("I feel terrible and sad")
        assert result.valence < 0

    def test_all_emotions_present(self):
        """Test that all 7 emotions are returned in results."""
        result = self.analyzer.analyze("Hello world")
        assert len(result.all_emotions) == 7
        expected_emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
        for emotion in expected_emotions:
            assert emotion in result.all_emotions

    def test_confidence_in_range(self):
        """Test that confidence is between 0 and 1."""
        result = self.analyzer.analyze("I am feeling various emotions")
        assert 0 <= result.confidence <= 1

    def test_arousal_in_range(self):
        """Test that arousal is between 0 and 1."""
        result = self.analyzer.analyze("I am feeling energetic")
        assert 0 <= result.arousal <= 1


class TestImageGeneratorDetection:
    """Test suite for ImageGenerator smart detection."""

    def setup_method(self):
        """Initialize generator before each test."""
        from src.services.image_generator import ImageGenerator
        self.generator = ImageGenerator()

    def test_emotional_input_detected(self):
        """Test that emotional inputs are correctly detected."""
        assert self.generator.is_emotional_input("I feel happy") == True
        assert self.generator.is_emotional_input("I am sad today") == True
        assert self.generator.is_emotional_input("feeling nervous") == True

    def test_object_input_detected(self):
        """Test that object inputs are correctly detected."""
        assert self.generator.is_emotional_input("red car") == False
        assert self.generator.is_emotional_input("mountain landscape") == False
        assert self.generator.is_emotional_input("coffee cup") == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
