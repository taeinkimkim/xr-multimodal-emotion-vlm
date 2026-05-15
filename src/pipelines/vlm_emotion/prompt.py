"""Prompt templates for VLM emotion experiments."""

from __future__ import annotations

DIRECT_PROMPTS: dict[int, str] = {
    1: """\
Look at this facial image and identify the person's emotion.

Choose exactly one emotion from: Happiness, Sadness, Anger, Surprise, Fear, Disgust, Neutral.

Reply in this exact format:
Emotion: <label>
Reasoning: <2–3 sentences describing the facial features you observed>""",

    2: """\
Analyze the facial image step by step:
1. Observation: Describe the specific facial features you see \
(eyebrows shape/position, eye openness, mouth corners, cheek tension, etc.)
2. Interpretation: Map those features to emotional signals. \
If multiple emotions are plausible, explain why you favor one over others.
3. Conclusion: State your final prediction.

Choose exactly one emotion from: Happiness, Sadness, Anger, Surprise, Fear, Disgust, Neutral.

Reply in this exact format:
Emotion: <label>
Observation: <facial features you noticed>
Reasoning: <how those features led to your choice, and why you ruled out alternatives>""",
}

ASSISTED_PROMPTS: dict[int, str] = {
    1: """\
A vision model has analyzed this facial image and provided the following context:
- Predicted emotion: {vision_label}
- Feature summary: {feature_summary}

Using this context alongside your own analysis, identify the most likely emotion.

Choose exactly one emotion from: Happiness, Sadness, Anger, Surprise, Fear, Disgust, Neutral.

Reply in this exact format:
Emotion: <label>
Reasoning: <2–3 sentences describing your analysis>""",

    2: """\
A separate vision model has analyzed the same image.

Vision model output:
- Predicted emotion: {vision_label}
- Feature summary: {feature_summary}

Analyze step by step:
1. Independent analysis: Based solely on what you see in the image, \
what emotion would you predict and why? Describe the visual cues.
2. Comparison: Does your independent prediction agree or disagree \
with the vision model's prediction ({vision_label})?
   - If agree: explain what facial features and vision model activations \
mutually reinforce this conclusion.
   - If disagree: explain what caused the discrepancy, evaluate which \
evidence is stronger, and justify your final choice.
3. Conclusion: State your final emotion.

Choose exactly one emotion from: Happiness, Sadness, Anger, Surprise, Fear, Disgust, Neutral.

Reply in this exact format:
Emotion: <label>
My independent prediction: <what you would predict without the vision model>
Agreement: <agree / disagree>
Reasoning: <detailed thought process — how the vision model's prediction and \
feature summary influenced or did not influence your final answer>""",
}

DIRECT_PROMPT_IDS   = sorted(DIRECT_PROMPTS.keys())
ASSISTED_PROMPT_IDS = sorted(ASSISTED_PROMPTS.keys())
