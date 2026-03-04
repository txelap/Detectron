class EvidenceSearcher:
    """
    Mock implementation of an Evidence Searcher.
    In a real system, this would query a search engine API (like Google Custom Search)
    or fact-checking databases (like Snopes, FactCheck.org) to find if a claim is verified.
    """
    def __init__(self):
        # Known controversial or false claims for mocking
        self.known_false_claims = [
            "virus se esconde en el agua",
            "vacuna contiene microchips",
            "tierra es plana"
        ]

    def analyze(self, headline: str, content: str) -> dict:
        """
        Simulates searching for evidence.
        Returns a score from 0.0 (strong evidence supporting claim)
        to 1.0 (strong evidence refuting claim or zero evidence found for a big claim).
        """
        text = (headline + " " + content).lower()

        # Check if the claim matches known falsehoods
        if any(claim in text for claim in self.known_false_claims):
            return {
                "score": 0.9,
                "is_verified": False,
                "consensus": "Refuted by major sources",
                "details": "Claim matches known debunked narratives."
            }

        # Simulate an unverified, unique claim (suspicious if big)
        if "secreto" in text or "revelación" in text:
            return {
                "score": 0.7,
                "is_verified": False,
                "consensus": "No secondary sources found",
                "details": "High-impact claim lacks corroboration from other outlets."
            }

        # Default to neutral/verified for standard news
        return {
            "score": 0.1,
            "is_verified": True,
            "consensus": "Corroborated by multiple sources",
            "details": "Found multiple reliable articles reporting similar facts."
        }
