from datetime import datetime

class ProfileAnalyzer:
    """
    Mock implementation of a Profile Analyzer.
    In a real system, this would analyze user metadata using tabular models
    like XGBoost and graph networks to identify bots or troll accounts.
    """
    def __init__(self):
        # In reality, this data would come from the social network's API (e.g., Twitter API)
        pass

    def analyze(self, user_profile: dict) -> dict:
        """
        Analyzes a user profile (metadata) for bot-like characteristics.

        Expected user_profile format:
        {
            "created_at": "YYYY-MM-DD",
            "followers_count": int,
            "following_count": int,
            "has_profile_pic": bool,
            "statuses_count": int
        }

        Returns a score from 0.0 (likely real human) to 1.0 (highly likely bot).
        """
        if not user_profile:
            return {
                "score": 0.5,
                "is_bot": False,
                "details": "No profile data provided. Defaulting to neutral."
            }

        score = 0.0
        details = []

        # 1. Profile Picture
        if not user_profile.get("has_profile_pic", True):
            score += 0.3
            details.append("Missing profile picture.")

        # 2. Account Age
        try:
            created_at = datetime.strptime(user_profile.get("created_at", "2000-01-01"), "%Y-%m-%d")
            age_days = (datetime.now() - created_at).days
            if age_days < 30:
                score += 0.4
                details.append("Account created recently (< 30 days).")
            elif age_days < 90:
                score += 0.2
                details.append("Account created recently (< 90 days).")
        except ValueError:
            pass

        # 3. Follower Ratio (High following, low followers = bot behavior)
        followers = max(user_profile.get("followers_count", 1), 1) # Avoid div by zero
        following = user_profile.get("following_count", 0)
        ratio = following / followers

        if ratio > 10.0 and following > 100:
            score += 0.4
            details.append(f"Suspicious following ratio ({following} following / {followers} followers).")

        # 4. Activity Level (Super high post count on a new account)
        statuses = user_profile.get("statuses_count", 0)
        if age_days and age_days > 0 and (statuses / age_days) > 100:
            score += 0.5
            details.append("Anomalous posting frequency (>100 posts/day).")

        final_score = min(1.0, score)
        return {
            "score": final_score,
            "is_bot": final_score > 0.6,
            "details": " ".join(details) if details else "Profile appears normal."
        }
