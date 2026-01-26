"""
NovaVision - Emotion to Image AI Generator
Author: Urme Bose (urme-b)
License: MIT
Repository: https://github.com/urme-b/NovaVision

Image Generator Service
=======================
This module provides high-quality image generation capabilities using the FLUX.1 model
through HuggingFace's Inference API. It transforms text descriptions and emotional
context into stunning AI-generated artwork.

Model Used:
    black-forest-labs/FLUX.1-dev
    - State-of-the-art text-to-image diffusion model
    - Produces high-quality 1024x1024 images
    - Optimized for photorealism and artistic styles
    - Fallback: FLUX.1-schnell for faster generation

Key Features:
    - Smart Input Detection: Distinguishes emotional inputs from object descriptions
    - Style Presets: Five curated styles (photorealistic, artistic, abstract, nature, dreamscape)
    - Emotion-to-Scene Mapping: Converts emotions to evocative visual scenes
    - Anti-Watermark Techniques: Includes tokens to minimize watermarks in output
    - Quality Modifiers: Automatic enhancement prompts for professional results

HuggingFace Inference API:
    - Requires HF_TOKEN environment variable
    - Supports async and sync generation
    - Rate limits apply based on account tier
"""

# Standard library imports
import os  # Environment variable access
import re  # Regular expressions for pattern matching

# Third-party imports
from dataclasses import dataclass  # Structured data containers
from huggingface_hub import InferenceClient  # HuggingFace API client for model inference
from PIL import Image  # Python Imaging Library for image handling
from dotenv import load_dotenv  # Environment variable loading from .env files

# Local imports
from src.services.emotion_analyzer import EmotionResult  # Emotion analysis result type

# Load environment variables from .env file
load_dotenv()


@dataclass
class GenerationResult:
    """
    Container for image generation results.

    This dataclass holds all information about a generated image, including
    the image itself, the prompt used, and metadata about the generation.

    Attributes:
        image: PIL Image object containing the generated artwork.
        prompt: The full enhanced prompt sent to the model.
        emotion: The primary emotion detected in the input.
        style: The visual style preset used for generation.
        input_type: Either 'object' (direct description) or 'emotion' (emotional input).
        seed: Random seed used for generation (for reproducibility).

    Example:
        >>> result = GenerationResult(
        ...     image=pil_image,
        ...     prompt="A serene meadow...",
        ...     emotion="joy",
        ...     style="nature",
        ...     input_type="emotion",
        ...     seed=12345
        ... )
    """
    image: Image.Image
    prompt: str
    emotion: str
    style: str
    input_type: str  # 'object' or 'emotion'
    seed: int  # The seed used for generation


# Emotion Detection Keywords
# ==========================
# These keywords help identify whether user input describes an emotional state
# versus a concrete object or scene. This distinction is crucial for prompt
# building - emotional inputs get mapped to mood-inspired scenes, while
# object inputs are rendered directly.
EMOTION_KEYWORDS = [
    # Direct emotion words
    'feel', 'feeling', 'felt', 'mood', 'emotion', 'emotional',
    # Positive emotions
    'happy', 'sad', 'angry', 'anxious', 'excited', 'calm', 'peaceful',
    # Negative emotions
    'stressed', 'love', 'hate', 'fear', 'joy', 'joyful', 'depressed',
    # Complex emotions
    'hopeful', 'worried', 'nervous', 'content', 'miserable', 'thrilled',
    'frustrated', 'annoyed', 'upset', 'grateful', 'proud', 'ashamed',
    'lonely', 'scared', 'terrified', 'delighted', 'furious', 'hurt',
    # First-person phrases indicating emotional expression
    "i'm", "i am", "i feel", "feeling like", "makes me feel"
]


# Style Presets
# =============
# Each preset contains carefully crafted prompt modifiers optimized for FLUX.1-dev.
# These modifiers guide the model toward specific visual aesthetics and quality levels.
STYLE_PRESETS = {
    # Photorealistic: Mimics high-end photography with realistic lighting and detail
    "photorealistic": (
        "hyperrealistic photograph, shot on Canon EOS R5, 85mm f/1.4 lens, "
        "professional studio lighting, DSLR quality, ultra sharp, lifelike, "
        "natural skin texture, ray tracing, ambient occlusion, 8k UHD resolution"
    ),
    # Artistic: Digital art style inspired by popular concept artists
    "artistic": (
        "masterpiece digital artwork, trending on artstation, highly detailed, "
        "concept art by greg rutkowski and alphonse mucha, smooth gradients, "
        "vibrant colors, dramatic lighting, illustration, award-winning art"
    ),
    # Abstract: Modern art with geometric and expressive elements
    "abstract": (
        "abstract modern art, geometric shapes, bold contrasting colors, "
        "contemporary gallery art, minimalist composition, artistic expression, "
        "museum quality, fine art print"
    ),
    # Nature: Wildlife and landscape photography aesthetic
    "nature": (
        "professional nature photography, National Geographic quality, "
        "golden hour cinematic lighting, ultra detailed landscape, "
        "shot on Hasselblad, vivid colors, atmospheric perspective, 8k"
    ),
    # Dreamscape: Surreal and fantasy-inspired imagery
    "dreamscape": (
        "surreal fantasy dreamscape, ethereal glowing atmosphere, "
        "digital art masterpiece, trending on artstation, concept art, "
        "imaginative world, volumetric lighting, magical realism"
    )
}


# Emotion-to-Scene Mappings
# =========================
# For emotional inputs, these mappings translate feelings into evocative visual
# scenes that capture the essence of each emotion. Used only when input is
# detected as emotional (not object/scene descriptions).
EMOTION_SCENES = {
    "joy": "radiant golden sunlit meadow with blooming wildflowers, warm summer day, happiness",
    "sadness": "misty rain-soaked forest at twilight, melancholic atmosphere, soft blue tones",
    "anger": "dramatic thunderstorm with lightning strikes, intense red and orange sky, powerful",
    "fear": "mysterious fog-shrouded landscape, dark shadows, eerie pale moonlight, suspense",
    "surprise": "spectacular burst of colorful aurora lights, electric energy, wonder",
    "disgust": "abstract organic textures, murky swamp atmosphere, unsettling greens",
    "neutral": "peaceful zen garden at dawn, balanced composition, serene tranquility"
}


# Quality Modifiers
# =================
# Base quality tokens that improve overall image quality across all styles.
# These are appended to every prompt to ensure professional-grade output.
QUALITY_BASE = (
    "masterpiece, best quality, ultra high definition, extremely detailed, "
    "sharp focus, professional, intricate details"
)


# Realism Enhancers
# =================
# Additional tokens that push the model toward more realistic rendering.
# Particularly effective for photorealistic and nature styles.
REALISM_ENHANCERS = (
    "hyperrealistic, photorealistic, lifelike, volumetric lighting, "
    "subsurface scattering, ray tracing, ambient occlusion, 8k resolution"
)


# Anti-Watermark Tokens
# =====================
# These tokens help minimize watermarks, logos, and text in generated images.
# FLUX models can sometimes produce artifacts resembling watermarks; these
# negative cues help reduce that tendency.
ANTI_WATERMARK = "clean image, no watermark, no text, no logo, no signature, no banner"


# Negative Prompt
# ===============
# Elements to explicitly avoid in generation. The model is instructed NOT to
# include these features, improving output quality and cleanliness.
NEGATIVE_PROMPT = (
    "watermark, logo, text, signature, letters, words, writing, brand, stamp, "
    "overlay, banner, copyright, trademark, username, website, url, "
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "cropped, out of frame, duplicate, error, jpeg artifacts, lowres"
)


class ImageGenerator:
    """
    High-quality image generation service using FLUX.1 models.

    This class handles the complete image generation pipeline:
    1. Detects whether input is emotional or descriptive
    2. Builds optimized prompts with style presets and quality modifiers
    3. Calls the HuggingFace Inference API for generation
    4. Returns structured results with metadata

    The generator uses FLUX.1-dev for primary generation (higher quality, ~10-20s)
    with automatic fallback to FLUX.1-schnell if the primary model fails.

    Attributes:
        hf_token: HuggingFace API token for authentication.
        model_id: The primary model identifier for generation.
        client: HuggingFace InferenceClient instance.

    Example:
        >>> generator = ImageGenerator()
        >>> result = generator.generate(
        ...     text="A peaceful morning",
        ...     emotion_result=emotion_result,
        ...     style="nature"
        ... )
        >>> result.image.save("output.png")

    Requires:
        HF_TOKEN environment variable set with valid HuggingFace API token.
    """

    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-dev"):
        """
        Initialize the image generator with FLUX.1-dev model.

        Args:
            model_id: HuggingFace model identifier for image generation.
                     Defaults to FLUX.1-dev for highest quality output.

        Raises:
            ValueError: If HF_TOKEN environment variable is not set.

        Example:
            >>> generator = ImageGenerator()  # Uses default FLUX.1-dev
            >>> generator = ImageGenerator("stabilityai/sdxl-turbo")  # Custom model
        """
        # Retrieve HuggingFace token from environment
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable not set")

        print(f"[ImageGenerator] Initializing with model: {model_id}")
        self.model_id = model_id

        # Initialize the HuggingFace Inference Client
        self.client = InferenceClient(token=self.hf_token)
        print("[ImageGenerator] Client initialized successfully")

    def is_emotional_input(self, text: str) -> bool:
        """
        Detect if user input describes emotions vs objects/scenes.

        This smart detection determines how the prompt should be built:
        - Emotional inputs: Get mapped to mood-inspired scenes
        - Object/scene inputs: Get rendered directly as described

        Detection uses multiple heuristics:
        1. Presence of emotion keywords
        2. First-person emotional phrases (I feel, I am, etc.)
        3. Input length (short inputs without emotion words = likely objects)

        Args:
            text: User input text to classify.

        Returns:
            True if input appears to describe emotions/feelings.
            False if input appears to describe objects/scenes.

        Example:
            >>> generator.is_emotional_input("I feel peaceful")
            True
            >>> generator.is_emotional_input("A red sports car")
            False
            >>> generator.is_emotional_input("cricket ball")
            False
        """
        text_lower = text.lower().strip()

        # Check for explicit emotion keywords
        for keyword in EMOTION_KEYWORDS:
            if keyword in text_lower:
                return True

        # Check for first-person emotional statements using regex
        # Matches patterns like "I feel sad", "We are happy", "My mood is..."
        if re.search(r'\b(i|we|my|me)\b.*\b(feel|am|was|been)\b', text_lower):
            return True

        # Short inputs without emotion words are likely object descriptions
        # e.g., "cat", "red car", "mountain lake"
        word_count = len(text_lower.split())
        if word_count <= 3:
            return False

        # Default to non-emotional for ambiguous cases
        return False

    def build_prompt(self, text: str, emotion_result: EmotionResult, style: str = "photorealistic") -> tuple[str, str]:
        """
        Build high-quality prompt optimized for FLUX.1-dev.

        This method constructs the full prompt sent to the model by:
        1. Detecting input type (emotional vs object)
        2. For emotional inputs: mapping to evocative scenes
        3. For object inputs: preserving the original description
        4. Adding style presets and quality modifiers
        5. Including anti-watermark tokens

        Prompt Engineering Strategy:
        - Emotional inputs get transformed into scene descriptions that
          capture the mood while adding atmospheric details
        - Object inputs are preserved verbatim but enhanced with quality
          modifiers to ensure professional output

        Args:
            text: Original user input text.
            emotion_result: EmotionResult from emotion analysis.
            style: Visual style preset key (photorealistic, artistic, etc.).

        Returns:
            Tuple of (full_prompt, input_type) where:
            - full_prompt: Complete prompt string for the model
            - input_type: 'emotion' or 'object' indicating detection result

        Example:
            >>> prompt, input_type = generator.build_prompt(
            ...     "I feel happy",
            ...     emotion_result,
            ...     "nature"
            ... )
            >>> print(input_type)
            'emotion'
        """
        # Get the style description for the selected preset
        style_desc = STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["photorealistic"])

        # Determine if this is an emotional or object-based input
        is_emotional = self.is_emotional_input(text)

        if is_emotional:
            # EMOTIONAL INPUT: Generate scene inspired by the detected emotion
            input_type = "emotion"
            emotion = emotion_result.primary_emotion

            # Get the scene template for this emotion
            emotion_scene = EMOTION_SCENES.get(emotion, EMOTION_SCENES["neutral"])

            # Build prompt: emotion scene + reference to original text + style + quality
            prompt = (
                f"{emotion_scene}, inspired by the feeling of '{text}', "
                f"{style_desc}, {QUALITY_BASE}, {REALISM_ENHANCERS}, "
                f"{ANTI_WATERMARK}, atmospheric, evocative, cinematic"
            )
        else:
            # OBJECT/SCENE INPUT: Generate exactly what user described
            input_type = "object"

            # For photorealistic style, add extra realism tokens
            if style.lower() == "photorealistic":
                prompt = (
                    f"{text}, {style_desc}, {QUALITY_BASE}, {REALISM_ENHANCERS}, "
                    f"{ANTI_WATERMARK}, award-winning photography, cinematic composition"
                )
            else:
                # Other styles get standard quality enhancement
                prompt = (
                    f"{text}, {style_desc}, {QUALITY_BASE}, "
                    f"{ANTI_WATERMARK}, highly detailed, professional quality"
                )

        # Log the prompt construction details
        print("=" * 70)
        print(f"[ImageGenerator] INPUT TYPE: {input_type.upper()}")
        print(f"[ImageGenerator] User input: '{text}'")
        print(f"[ImageGenerator] Style: {style}")
        print(f"[ImageGenerator] FINAL PROMPT ({len(prompt)} chars):")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("=" * 70)

        return prompt, input_type

    def generate(
        self,
        text: str,
        emotion_result: EmotionResult,
        style: str = "photorealistic",
        width: int = 1024,
        height: int = 1024,
        seed: int = None
    ) -> GenerationResult:
        """
        Generate high-quality image using FLUX.1-dev.

        This is the main generation method that orchestrates the full pipeline:
        1. Generates random seed if not provided (for reproducibility)
        2. Builds optimized prompt using smart detection
        3. Calls HuggingFace Inference API with FLUX.1-dev
        4. Handles errors with automatic retry and fallback

        Generation Parameters:
        - FLUX.1-dev uses 30 inference steps for high quality
        - Guidance scale of 3.5 is optimal for FLUX.1-dev
        - Fallback to FLUX.1-schnell (4 steps) if primary fails

        Args:
            text: User input text describing desired image or emotion.
            emotion_result: EmotionResult from emotion analysis.
            style: Visual style preset (default: "photorealistic").
            width: Output image width in pixels (default: 1024).
            height: Output image height in pixels (default: 1024).
            seed: Random seed for reproducibility (default: random).

        Returns:
            GenerationResult containing the generated image and metadata.

        Raises:
            Exception: If all generation attempts fail (primary + retry + fallback).

        Example:
            >>> result = generator.generate(
            ...     text="A majestic eagle soaring",
            ...     emotion_result=emotion_result,
            ...     style="nature",
            ...     seed=42
            ... )
            >>> result.image.save("eagle.png")
            >>> print(f"Seed used: {result.seed}")
        """
        import random

        # Generate random seed if not provided (allows reproducibility)
        if seed is None:
            seed = random.randint(0, 2147483647)

        # Log generation parameters
        print(f"\n[ImageGenerator] Processing request...")
        print(f"[ImageGenerator] User text: '{text}'")
        print(f"[ImageGenerator] Style: {style}, Size: {width}x{height}")
        print(f"[ImageGenerator] Seed: {seed}")
        print(f"[ImageGenerator] Model: {self.model_id}")

        # Build the optimized prompt with smart detection
        prompt, input_type = self.build_prompt(text, emotion_result, style)

        # Generate image using HuggingFace Inference API
        print("[ImageGenerator] Calling HuggingFace API (this may take 10-20 seconds)...")

        try:
            # Primary attempt: FLUX.1-dev with full parameters
            # 30 inference steps provides high quality at ~10-20s generation time
            image = self.client.text_to_image(
                prompt=prompt,
                model=self.model_id,
                width=width,
                height=height,
                num_inference_steps=30,    # More steps = better quality
                guidance_scale=3.5,        # Optimal for FLUX.1-dev
                negative_prompt=NEGATIVE_PROMPT,
            )
            print("[ImageGenerator] Image generated successfully!")

        except Exception as e:
            # First retry: Use simplified parameters
            print(f"[ImageGenerator] Error with full params: {e}")
            print("[ImageGenerator] Retrying with basic parameters...")

            try:
                image = self.client.text_to_image(
                    prompt=prompt,
                    model=self.model_id,
                    width=width,
                    height=height,
                    num_inference_steps=25,
                )
                print("[ImageGenerator] Retry successful!")

            except Exception as retry_error:
                # Final fallback: Use FLUX.1-schnell (faster but lower quality)
                print(f"[ImageGenerator] Retry failed: {retry_error}")
                print("[ImageGenerator] Final attempt with FLUX.1-schnell fallback...")

                try:
                    image = self.client.text_to_image(
                        prompt=prompt,
                        model="black-forest-labs/FLUX.1-schnell",
                        width=width,
                        height=height,
                        num_inference_steps=4,  # Schnell uses fewer steps
                    )
                    print("[ImageGenerator] Fallback to schnell successful!")
                except Exception as final_error:
                    print(f"[ImageGenerator] All attempts failed: {final_error}")
                    raise

        # Return structured result with all metadata
        return GenerationResult(
            image=image,
            prompt=prompt,
            emotion=emotion_result.primary_emotion,
            style=style,
            input_type=input_type,
            seed=seed
        )


# Module self-test - run when executed directly
if __name__ == "__main__":
    from src.services.emotion_analyzer import EmotionAnalyzer

    print("=" * 70)
    print("Testing ImageGenerator with FLUX.1-dev")
    print("=" * 70)

    analyzer = EmotionAnalyzer()
    generator = ImageGenerator()

    # Test cases covering different input types and styles
    test_cases = [
        ("cricket ball", "photorealistic"),       # Object input
        ("virat playing cricket", "photorealistic"),  # Scene input
        ("I feel happy and excited", "artistic"),  # Emotional input
    ]

    for text, style in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: '{text}' with {style} style")
        print(f"{'='*70}")

        # Check input type detection
        is_emotional = generator.is_emotional_input(text)
        print(f"Is emotional input: {is_emotional}")

        # Analyze emotion
        emotion_result = analyzer.analyze(text)
        print(f"Detected emotion: {emotion_result.primary_emotion}")

        # Build prompt (without generating image in test)
        prompt, input_type = generator.build_prompt(text, emotion_result, style)
        print(f"Input type: {input_type}")
