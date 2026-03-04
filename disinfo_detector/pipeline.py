import json
from modules.nlp import ClickbaitDetector, EmotionAnalyzer
from modules.source import SourceAnalyzer, DateVerifier
from modules.evidence import EvidenceSearcher
from modules.profile import ProfileAnalyzer
from modules.media import MediaAnalyzer

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
        self.evidence_searcher = EvidenceSearcher()
        self.profile_analyzer = ProfileAnalyzer()
        self.media_analyzer = MediaAnalyzer()

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

        # 5. Search Evidence
        evidence_result = self.evidence_searcher.analyze(
            post_data.get("headline", ""),
            post_data.get("content", "")
        )

        # 6. Analyze Profile
        profile_result = self.profile_analyzer.analyze(
            post_data.get("user_profile", {})
        )

        # 7. Analyze Media
        media_result = self.media_analyzer.analyze(
            post_data.get("media_url", "")
        )

        # 8. Calculate Final Trust Score
        # A simple weighted average for the mock combining all 7 modules.
        # Higher score means MORE likely to be disinformation.

        # Source score is reliability (0-1), so we invert it to get "unreliability"
        unreliability_score = 1.0 - source_result["score"]

        # Weights (total = 1.0)
        weights = {
            "source": 0.20,
            "clickbait": 0.10,
            "emotion": 0.10,
            "date": 0.10,
            "evidence": 0.25,
            "profile": 0.10,
            "media": 0.15
        }

        disinfo_score = (
            (unreliability_score * weights["source"]) +
            (clickbait_result["score"] * weights["clickbait"]) +
            (emotion_result["score"] * weights["emotion"]) +
            (date_result["score"] * weights["date"]) +
            (evidence_result["score"] * weights["evidence"]) +
            (profile_result["score"] * weights["profile"]) +
            (media_result["score"] * weights["media"])
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
                "1_source_reliability": source_result,
                "2_clickbait_analysis": clickbait_result,
                "3_emotion_analysis": emotion_result,
                "4_date_verification": date_result,
                "5_evidence_search": evidence_result,
                "6_profile_analysis": profile_result,
                "7_media_analysis": media_result
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
