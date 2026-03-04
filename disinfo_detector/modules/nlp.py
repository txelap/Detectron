import re

class ClickbaitDetector:
    """
    Mock implementation of a Clickbait Detector.
    In a real system, this would use an NLI (Natural Language Inference) model
    like RoBERTa to compare the headline with the article body.
    """
    def __init__(self):
        # Common clickbait patterns
        self.clickbait_patterns = [
            r"\b(no creerás|increíble|impactante|lo que pasó|te sorprenderá|el secreto|la verdad sobre)\b",
            r"\!{2,}",  # Multiple exclamation marks
            r"\?{2,}"   # Multiple question marks
        ]

    def analyze(self, headline: str, content: str) -> dict:
        """
        Analyzes the headline and content for clickbait indicators.
        Returns a score from 0.0 (legitimate) to 1.0 (highly likely clickbait).
        """
        score = 0.0
        headline_lower = headline.lower()

        # Check heuristics
        for pattern in self.clickbait_patterns:
            if re.search(pattern, headline_lower):
                score += 0.3

        # Mock logic: If content is very short compared to headline, might be clickbait
        if len(content) < len(headline) * 2:
            score += 0.4

        # Cap score at 1.0
        final_score = min(1.0, score)

        return {
            "score": final_score,
            "is_clickbait": final_score > 0.6,
            "details": "Heuristic analysis (mock NLI)"
        }


class EmotionAnalyzer:
    """
    Mock implementation of an Emotion Analyzer.
    In a real system, this would use a transformer trained on emotion datasets
    to detect fear, anger, joy, sadness, etc.
    """
    def __init__(self):
        # Simple keywords mapping to strong emotions
        self.fear_keywords = ['terror', 'pánico', 'peligro', 'amenaza', 'destrucción', 'virus', 'mortal']
        self.anger_keywords = ['indignante', 'corrupción', 'robo', 'furia', 'odio', 'asco', 'vergüenza']

    def analyze(self, text: str) -> dict:
        """
        Analyzes the text for manipulative emotions (fear, anger).
        Returns a score from 0.0 (neutral) to 1.0 (highly emotionally manipulative).
        """
        text_lower = text.lower()
        words = text_lower.split()

        fear_count = sum(1 for word in words if any(kw in word for kw in self.fear_keywords))
        anger_count = sum(1 for word in words if any(kw in word for kw in self.anger_keywords))

        # Simple scoring based on word frequency
        total_emotion_words = fear_count + anger_count

        # Arbitrary scaling for the mock
        score = min(1.0, total_emotion_words * 0.15)

        return {
            "score": score,
            "dominant_emotion": "fear" if fear_count > anger_count else ("anger" if anger_count > 0 else "neutral"),
            "details": "Keyword-based emotion detection (mock Transformer)"
        }
