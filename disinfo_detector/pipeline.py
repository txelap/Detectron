import json
from modules.nlp import ClickbaitDetector, EmotionAnalyzer
from modules.source import SourceAnalyzer, DateVerifier

class DisinformationPipeline:
    """
    The main pipeline that orchestrates the different analysis modules
    to evaluate a social media post or news article for disinformation.
    """
    def __init__(self):
        # Initialize modules
        self.clickbait_detector = ClickbaitDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.source_analyzer = SourceAnalyzer()
        self.date_verifier = DateVerifier()

    def analyze_post(self, post_data: dict) -> dict:
        """
        Runs the full analysis pipeline on a given post.

        Expected post_data format:
        {
            "headline": str,
            "content": str,
            "source_url": str,
            "reported_date": "YYYY-MM-DD"
        }
        """
        # 1. Analyze Source
        source_result = self.source_analyzer.analyze(post_data.get("source_url", ""))

        # 2. Detect Clickbait
        clickbait_result = self.clickbait_detector.analyze(
            post_data.get("headline", ""),
            post_data.get("content", "")
        )

        # 3. Analyze Emotion
        emotion_result = self.emotion_analyzer.analyze(
            post_data.get("headline", "") + " " + post_data.get("content", "")
        )

        # 4. Verify Date
        date_result = self.date_verifier.analyze(
            post_data.get("reported_date", "")
        )

        # 5. Calculate Final Trust Score
        # A simple weighted average for the mock.
        # Higher score means MORE likely to be disinformation.

        # Source score is reliability (0-1), so we invert it to get "unreliability"
        unreliability_score = 1.0 - source_result["score"]

        # Weights
        w_source = 0.4
        w_clickbait = 0.2
        w_emotion = 0.2
        w_date = 0.2

        disinfo_score = (
            (unreliability_score * w_source) +
            (clickbait_result["score"] * w_clickbait) +
            (emotion_result["score"] * w_emotion) +
            (date_result["score"] * w_date)
        )

        # Ensure score is between 0 and 1
        disinfo_score = max(0.0, min(1.0, disinfo_score))

        final_assessment = "Trustworthy"
        if disinfo_score > 0.7:
            final_assessment = "High Risk of Disinformation"
        elif disinfo_score > 0.4:
            final_assessment = "Suspicious - Verify Sources"

        # Compile results
        report = {
            "post_summary": {
                "headline": post_data.get("headline", ""),
                "domain": source_result.get("domain", "Unknown")
            },
            "analysis": {
                "source_reliability": source_result,
                "clickbait_analysis": clickbait_result,
                "emotion_analysis": emotion_result,
                "date_verification": date_result
            },
            "final_score": {
                "disinformation_probability": round(disinfo_score, 2),
                "assessment": final_assessment
            }
        }

        return report

    def print_report(self, report: dict):
        """Helper to pretty-print the analysis report."""
        print(json.dumps(report, indent=4, ensure_ascii=False))
