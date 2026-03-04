import random

class MediaAnalyzer:
    """
    Mock implementation of a Media/Deepfake Analyzer.
    In a real system, this would load images and videos using CNNs
    (e.g., EfficientNet, Vision Transformers) to detect generative AI artifacts.
    """
    def __init__(self):
        # Known paths for mock tests
        pass

    def analyze(self, media_url: str) -> dict:
        """
        Analyzes media (image, video, audio) for deepfake patterns.
        Returns a score from 0.0 (authentic media) to 1.0 (highly likely AI-generated).
        """
        if not media_url:
            return {
                "score": 0.0,
                "is_deepfake": False,
                "media_type": "none",
                "details": "No media provided."
            }

        # Determine media type based on extension
        url_lower = media_url.lower()
        media_type = "unknown"
        if url_lower.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            media_type = "image"
        elif url_lower.endswith(('.mp4', '.avi', '.mov')):
            media_type = "video"
        elif url_lower.endswith(('.mp3', '.wav')):
            media_type = "audio"

        # Mock Analysis
        # 1. AI images often have specific noise patterns or mismatched shadows.
        if "ai-gen" in url_lower or "midjourney" in url_lower:
            score = 0.95
            details = f"Detected high-frequency noise typical of latent diffusion models in {media_type}."
        elif "deepfake" in url_lower:
            score = 0.98
            details = f"Facial artifacts and blending inconsistencies found in {media_type}."
        else:
            # Simulate analyzing a random real image
            # Provide a low baseline score indicating authenticity
            score = random.uniform(0.01, 0.15)
            details = f"No significant AI artifacts detected in {media_type}. Appears authentic."

        return {
            "score": round(score, 2),
            "is_deepfake": score > 0.6,
            "media_type": media_type,
            "details": details
        }
