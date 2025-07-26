from typing import Dict, List, Any
import torch
from transformers.pipelines import pipeline


class EmotionDetector:
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the emotion detection pipeline with the specified model.
        Automatically uses GPU if available.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,  # Get all class scores
            device=device
        )

    def detect(self, text: str) -> Dict[str, float]:
        """
        Analyze emotion in the given text and return scores per emotion label.

        :param text: The input string to analyze.
        :return: A dictionary mapping emotions to confidence scores.
        """
        try:
            # Preprocess input
            text = text.strip().replace("\n", " ")[:512]

            # Get raw predictions from pipeline
            raw_result: List[Dict[str, Any]] = self.pipeline(text)

            # Flatten in case model returns [[...]] instead of [...]
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], list):
                result = raw_result[0]
            else:
                result = raw_result

            # Parse emotion scores
            emotion_scores = {
                r["label"]: float(r["score"])
                for r in result
                if isinstance(r, dict) and "label" in r and "score" in r
            }

            return emotion_scores or {"neutral": 1.0}  # fallback default

        except Exception as e:
            print(f"[EmotionDetector] Error: {e}")
            return {"neutral": 1.0}
