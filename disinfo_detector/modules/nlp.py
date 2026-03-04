import re
import torch
from transformers import pipeline

class ClickbaitDetector:
    """
    Implementation of a Clickbait Detector.
    Uses a combination of heuristics and a real Zero-Shot Classification NLP model
    to detect if a headline is clickbait.
    """
    def __init__(self):
        # We use a lightweight zero-shot classification model from Hugging Face
        # to classify the headline into 'clickbait' or 'news'

        # Determine if a GPU is available (e.g., GTX 1060) to speed up inference
        device_id = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device_id == 0 else "CPU"

        print(f"Loading NLP Model for Clickbait Detection on {device_name}...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-distilroberta-base",
            device=device_id
        )

        self.clickbait_patterns = [
            r"\b(no creerás|increíble|impactante|lo que pasó|te sorprenderá|el secreto|la verdad sobre)\b",
            r"\!{2,}",  # Multiple exclamation marks
            r"\?{2,}"   # Multiple question marks
        ]

    def analyze(self, headline: str, content: str) -> dict:
        """
        Analyzes the headline for clickbait indicators using NLP.
        Returns a score from 0.0 (legitimate) to 1.0 (highly likely clickbait).
        """
        if not headline:
            return {"score": 0.0, "is_clickbait": False, "details": "No headline"}

        score = 0.0

        # 1. NLP Analysis
        try:
            result = self.classifier(headline, candidate_labels=["sensationalist clickbait", "factual news headline"])
            # Get the probability score of it being clickbait
            clickbait_idx = result['labels'].index("sensationalist clickbait")
            nlp_score = result['scores'][clickbait_idx]
            score += nlp_score * 0.7  # NLP carries 70% of the weight
        except Exception as e:
            print(f"NLP Clickbait error: {e}")
            nlp_score = 0.0

        # 2. Heuristics Backup (30% weight)
        headline_lower = headline.lower()
        heuristic_score = 0.0
        for pattern in self.clickbait_patterns:
            if re.search(pattern, headline_lower):
                heuristic_score += 0.5

        if len(content) < len(headline) * 2 and len(content) > 0:
            heuristic_score += 0.5

        score += min(1.0, heuristic_score) * 0.3

        final_score = min(1.0, score)

        return {
            "score": round(final_score, 2),
            "is_clickbait": final_score > 0.6,
            "details": f"AI NLP Clickbait Probability: {nlp_score:.2f}"
        }


class EmotionAnalyzer:
    """
    Implementation of an Emotion Analyzer.
    Uses a real Hugging Face sentiment analysis model to detect strong manipulative emotions.
    """
    def __init__(self):
        # Determine if a GPU is available to speed up inference
        device_id = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device_id == 0 else "CPU"

        print(f"Loading NLP Model for Emotion Detection on {device_name}...")

        # Using a multilingual sentiment model that detects negativity strongly
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=device_id
        )

    def analyze(self, text: str) -> dict:
        """
        Analyzes the text for manipulative emotions (negative/angry sentiment).
        Returns a score from 0.0 (neutral/positive) to 1.0 (highly emotionally negative/manipulative).
        """
        if not text:
            return {"score": 0.0, "dominant_emotion": "neutral", "details": "No text provided"}

        # Truncate text to fit model max length (usually 512 tokens, we take ~300 words)
        truncated_text = " ".join(text.split()[:300])

        try:
            result = self.analyzer(truncated_text)[0]
            label = result['label'] # Output like "1 star", "5 stars"
            confidence = result['score']

            # The model outputs 1 to 5 stars.
            # 1 and 2 stars are highly negative (anger, fear, outrage).
            # We map this to our manipulation score.
            if label == '1 star':
                score = 0.9 * confidence
                emotion = "anger/outrage"
            elif label == '2 stars':
                score = 0.7 * confidence
                emotion = "negative"
            elif label == '3 stars':
                score = 0.2
                emotion = "neutral"
            else:
                score = 0.0
                emotion = "positive"

        except Exception as e:
            print(f"NLP Emotion error: {e}")
            score = 0.0
            emotion = "unknown"
            confidence = 0.0

        return {
            "score": round(score, 2),
            "dominant_emotion": emotion,
            "details": f"AI Multilingual Sentiment: {label} (Confidence: {confidence:.2f})"
        }
