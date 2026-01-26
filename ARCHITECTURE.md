# System Architecture

## Overview
NovaVision transforms text into AI-generated art using emotion detection and diffusion models.

## Data Flow
```
User Input → Smart Detection → Emotion Analysis → Prompt Building → Image Generation → Output
```

## Pipeline Steps

**Step 1: Input**
- User enters text + selects style
- Example: "I feel happy today" + Artistic

**Step 2: Smart Detection**
- Checks if input is EMOTION ("I feel happy") or OBJECT ("red car")
- Routes to appropriate processing

**Step 3: Emotion Analysis** (if emotion mode)
- Model: emotion-english-distilroberta-base
- Output: primary emotion, confidence, valence, arousal

**Step 4: Prompt Building**
- Maps emotion → visual elements (colors, lighting, mood)
- Adds style modifiers + quality keywords
- Appends anti-watermark suffix

**Step 5: Image Generation**
- Model: FLUX.1-schnell via HuggingFace API
- Resolution: 1024x1024
- Returns PIL Image

## Emotion Mapping
| Emotion | Valence | Arousal | Visual Style |
|---------|---------|---------|--------------|
| Joy | +0.8 | 0.7 | Bright, warm, golden |
| Sadness | -0.7 | 0.3 | Muted, cool, dim |
| Anger | -0.6 | 0.9 | Red, sharp, intense |
| Fear | -0.8 | 0.8 | Dark, shadows, tense |
| Neutral | 0.0 | 0.3 | Balanced, calm |

## Style Presets
| Style | Modifiers |
|-------|-----------|
| Artistic | oil painting, brushwork, museum quality |
| Photorealistic | DSLR, 85mm lens, 8k, natural light |
| Abstract | geometric, bold colors, modern art |
| Nature | landscape, golden hour, serene |
| Dreamscape | surreal, fantasy, ethereal |

## Tech Stack
- **Frontend:** Gradio / HTML
- **Backend:** Python / Flask
- **NLP:** HuggingFace Transformers
- **Image:** FLUX.1 via Inference API
