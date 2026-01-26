"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Pipeline Orchestration Module
=============================
This module provides an end-to-end pipeline that combines emotion analysis
and image generation into a single, cohesive workflow. It serves as the
main orchestrator for the complete emotion-to-image transformation process.

Pipeline Steps:
    1. Emotion Analysis: Detect emotional content in user text
    2. Prompt Enhancement: Build optimized prompt based on emotion and style
    3. Image Generation: Generate artwork using the enhanced prompt

Architecture:
    The pipeline follows a sequential processing model where each step
    builds upon the results of the previous step. This design allows for
    easy extension and modification of individual components.

Note:
    This module is provided for alternative usage patterns. The main
    application (server.py and app.py) implements lightweight orchestration
    directly for better control over the generation process.
"""

# Local imports
from src.services.emotion_analyzer import EmotionAnalyzer  # Emotion analysis service
from src.services.image_generator import ImageGenerator, GenerationResult  # Image generation


class NovaVisionPipeline:
    """
    End-to-end pipeline for emotion-to-image generation.

    This class orchestrates the complete transformation from text input
    to generated artwork. It manages the interaction between the emotion
    analyzer and image generator services.

    The pipeline is designed for batch processing and provides a clean
    interface for running the full generation workflow with a single
    method call.

    Attributes:
        emotion_analyzer: Instance of EmotionAnalyzer for text analysis.
        image_generator: Instance of ImageGenerator for image creation.

    Example:
        >>> pipeline = NovaVisionPipeline()
        >>> result = pipeline.run("I feel peaceful", style="nature")
        >>> print(result.emotion)
        'joy'
        >>> result.image.save("output.png")

    Note:
        Requires HF_TOKEN environment variable to be set for image generation.
    """

    def __init__(self):
        """
        Initialize the pipeline with required services.

        Creates instances of the emotion analyzer and image generator.
        Model loading happens during this initialization, which may take
        a few seconds on first run.
        """
        self._analyzer = None
        self._generator = None

    @property
    def emotion_analyzer(self) -> EmotionAnalyzer:
        """Lazy-loaded emotion analyzer instance."""
        if self._analyzer is None:
            print("[Pipeline] Initializing EmotionAnalyzer...")
            self._analyzer = EmotionAnalyzer()
        return self._analyzer

    @property
    def image_generator(self) -> ImageGenerator:
        """Lazy-loaded image generator instance."""
        if self._generator is None:
            print("[Pipeline] Initializing ImageGenerator...")
            self._generator = ImageGenerator()
        return self._generator

    def run(
        self,
        text: str,
        style: str = "artistic",
        width: int = 1024,
        height: int = 1024,
        seed: int = None
    ) -> GenerationResult:
        """
        Run the full emotion-to-image pipeline.

        This method executes the complete transformation workflow:
        1. Analyzes the input text for emotional content
        2. Builds optimized prompt based on emotion and style
        3. Generates image using the enhanced prompt
        4. Returns structured result with all metadata

        Args:
            text: Input text to analyze and visualize.
            style: Visual style preset (default: "artistic").
                   Options: photorealistic, artistic, abstract, nature, dreamscape
            width: Output image width in pixels (default: 1024).
            height: Output image height in pixels (default: 1024).
            seed: Random seed for reproducibility (default: random).

        Returns:
            GenerationResult containing:
                - image: PIL Image object
                - prompt: The enhanced prompt used
                - emotion: Detected primary emotion
                - style: Style preset used
                - input_type: 'emotion' or 'object'
                - seed: The seed used for generation

        Raises:
            ValueError: If input text is empty or invalid.
            Exception: If image generation fails.

        Example:
            >>> pipeline = NovaVisionPipeline()
            >>> result = pipeline.run(
            ...     text="I feel excited about the future",
            ...     style="dreamscape"
            ... )
            >>> print(f"Emotion: {result.emotion}")
            Emotion: joy
            >>> result.image.save("excited.png")
        """
        # Step 1: Analyze the emotional content of the input text
        print(f"[Pipeline] Analyzing: '{text[:50]}...' " if len(text) > 50 else f"[Pipeline] Analyzing: '{text}'")
        emotion_result = self.emotion_analyzer.analyze(text)
        print(f"[Pipeline] Detected emotion: {emotion_result.primary_emotion}")

        # Step 2 & 3: Generate image (prompt building is handled internally)
        print(f"[Pipeline] Generating image with style: {style}")
        result = self.image_generator.generate(
            text=text,
            emotion_result=emotion_result,
            style=style,
            width=width,
            height=height,
            seed=seed
        )

        print(f"[Pipeline] Generation complete! Input type: {result.input_type}")
        return result


# Singleton Pattern Implementation
# ================================
# Module-level variable and getter function for lazy initialization
_pipeline: NovaVisionPipeline | None = None


def get_pipeline() -> NovaVisionPipeline:
    """
    Get or create the singleton pipeline instance.

    Implements lazy initialization to defer model loading until the
    pipeline is actually needed. Subsequent calls return the same instance.

    Returns:
        NovaVisionPipeline: The singleton pipeline instance.

    Example:
        >>> pipeline = get_pipeline()
        >>> # Same instance returned on subsequent calls
        >>> pipeline2 = get_pipeline()
        >>> pipeline is pipeline2
        True
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = NovaVisionPipeline()
    return _pipeline
