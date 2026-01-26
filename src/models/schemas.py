"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Pydantic Schema Definitions
===========================
This module defines Pydantic models for data validation and serialization
throughout the NovaVision application. These schemas ensure type safety
and provide clear API contracts for request/response handling.

Schema Types:
    - EmotionType: Enumeration of supported emotion categories
    - EmotionResult: Emotion analysis output structure
    - ImageGenerationRequest: API request structure for generation
    - GenerationResult: Complete pipeline output structure

Usage:
    from src.models.schemas import EmotionType, ImageGenerationRequest

    request = ImageGenerationRequest(
        text="I feel happy today",
        style="artistic"
    )
"""

# Standard library imports
from enum import Enum  # Enumeration support for emotion types
from typing import Optional  # Optional type hints

# Third-party imports
from pydantic import BaseModel, Field  # Pydantic for data validation


class EmotionType(str, Enum):
    """
    Enumeration of supported emotion categories.

    This enum defines the seven emotion categories detected by the
    emotion analysis model (j-hartmann/emotion-english-distilroberta-base).
    Each emotion represents a distinct affective state.

    Emotion Descriptions:
        ANGER: Frustration, irritation, or rage
        DISGUST: Revulsion or strong disapproval
        FEAR: Anxiety, worry, or dread
        JOY: Happiness, pleasure, or positive feelings
        NEUTRAL: Absence of strong emotional content
        SADNESS: Grief, sorrow, or melancholy
        SURPRISE: Astonishment, amazement, or unexpected reactions

    Example:
        >>> emotion = EmotionType.JOY
        >>> emotion.value
        'joy'
        >>> EmotionType("sadness")
        <EmotionType.SADNESS: 'sadness'>
    """
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    NEUTRAL = "neutral"
    SADNESS = "sadness"
    SURPRISE = "surprise"


class EmotionResult(BaseModel):
    """
    Pydantic model for emotion analysis results.

    This model validates and structures the output from emotion analysis,
    ensuring consistent data types and value ranges.

    Attributes:
        emotion: The primary detected emotion (EmotionType enum value).
        confidence: Confidence score for the primary emotion (0.0-1.0).
        all_scores: Dictionary mapping all emotion categories to their scores.

    Validation:
        - confidence must be between 0.0 and 1.0 (inclusive)
        - emotion must be a valid EmotionType value

    Example:
        >>> result = EmotionResult(
        ...     emotion=EmotionType.JOY,
        ...     confidence=0.87,
        ...     all_scores={"joy": 0.87, "neutral": 0.08, ...}
        ... )
    """
    emotion: EmotionType
    confidence: float = Field(..., ge=0.0, le=1.0)  # Constrained to [0, 1]
    all_scores: dict[str, float]


class ImageGenerationRequest(BaseModel):
    """
    Pydantic model for image generation API requests.

    This model validates incoming generation requests, ensuring all
    parameters are within acceptable ranges and providing sensible defaults.

    Attributes:
        text: The input text to analyze and visualize (1-1000 chars).
        style: Visual style preset (default: "artistic").
        num_images: Number of images to generate (1-4, default: 1).
        guidance_scale: How closely to follow the prompt (1.0-20.0, default: 7.5).
        num_inference_steps: Denoising steps (10-50, default: 30).

    Validation:
        - text must be 1-1000 characters
        - num_images must be 1-4
        - guidance_scale must be 1.0-20.0
        - num_inference_steps must be 10-50

    Example:
        >>> request = ImageGenerationRequest(
        ...     text="A peaceful sunset over the ocean",
        ...     style="nature",
        ...     guidance_scale=8.0
        ... )
    """
    text: str = Field(..., min_length=1, max_length=1000)
    style: Optional[str] = "artistic"
    num_images: int = Field(default=1, ge=1, le=4)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=30, ge=10, le=50)


class GenerationResult(BaseModel):
    """
    Pydantic model for complete pipeline output.

    This model structures the full output from the emotion-to-image
    generation pipeline, including all intermediate results and metadata.

    Attributes:
        original_text: The original user input text.
        detected_emotion: EmotionResult from the analysis phase.
        enhanced_prompt: The prompt sent to the image generation model.
        image_paths: List of paths to generated image files.

    Example:
        >>> result = GenerationResult(
        ...     original_text="I feel happy",
        ...     detected_emotion=emotion_result,
        ...     enhanced_prompt="radiant golden sunlit meadow...",
        ...     image_paths=["outputs/image_001.png"]
        ... )
    """
    original_text: str
    detected_emotion: EmotionResult
    enhanced_prompt: str
    image_paths: list[str]
