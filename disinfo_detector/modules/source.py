from datetime import datetime
import re

class SourceAnalyzer:
    """
    Mock implementation of a Source Analyzer.
    In a real system, this would check against a comprehensive database of news sources,
    fact-checking sites, and look-alike domains.
    """
    def __init__(self):
        # A simple list of domains considered generally reliable (for demonstration purposes only)
        self.reliable_domains = [
            'bbc.com', 'reuters.com', 'apnews.com', 'elpais.com', 'nytimes.com', 'cnn.com'
        ]
        # Known satire or known fake news domains
        self.unreliable_domains = [
            'elmundotoday.com', 'theonion.com', 'noticiasfalsas.com', 'clickbait-news.net'
        ]

    def extract_domain(self, url: str) -> str:
        """Extracts the domain from a URL using a simple regex."""
        match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if match:
            return match.group(1).lower()
        return ""

    def analyze(self, source_url: str) -> dict:
        """
        Analyzes the source URL.
        Returns a reliability score from 0.0 (unreliable/unknown) to 1.0 (highly reliable).
        """
        domain = self.extract_domain(source_url)

        if domain in self.reliable_domains:
            return {
                "score": 0.9,
                "is_reliable": True,
                "domain": domain,
                "details": "Source is in the reliable domains list."
            }
        elif domain in self.unreliable_domains:
            return {
                "score": 0.1,
                "is_reliable": False,
                "domain": domain,
                "details": "Source is known to be satire or unreliable."
            }
        else:
            # Unknown domain
            return {
                "score": 0.4,
                "is_reliable": False,
                "domain": domain,
                "details": "Unknown source domain. Requires cross-verification."
            }


class DateVerifier:
    """
    Mock implementation of a Date Verifier.
    In a real system, this would extract dates from the text using NER and compare
    them against the current date to see if old news is being recycled.
    """
    def __init__(self):
        pass

    def analyze(self, reported_date_str: str, content: str = "") -> dict:
        """
        Analyzes the reported date. In a full system, 'content' would be parsed for dates.
        Returns a score from 0.0 (current/relevant) to 1.0 (highly likely recycled/old).
        """
        try:
            # Assuming format 'YYYY-MM-DD' for simplicity in the mock
            reported_date = datetime.strptime(reported_date_str, "%Y-%m-%d")
            current_date = datetime.now()

            delta = current_date - reported_date

            # If the news is more than 30 days old, flag it as potentially recycled
            # In a real system, this would depend on the context of the news.
            days_old = delta.days

            if days_old > 365:
                score = 0.9  # Very old news
            elif days_old > 30:
                score = 0.6  # Old news
            elif days_old < 0:
                score = 1.0  # Future date? Suspicious!
            else:
                score = 0.1  # Recent news

            return {
                "score": score,
                "days_old": days_old,
                "is_recycled": score > 0.5,
                "details": f"News is approximately {days_old} days old."
            }

        except ValueError:
             return {
                "score": 0.5,
                "days_old": None,
                "is_recycled": False,
                "details": "Could not parse date format."
            }
