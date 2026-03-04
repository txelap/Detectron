import json
from datetime import datetime
import feedparser
from bs4 import BeautifulSoup
import time

class RealWorldDataIngestion:
    """
    Connects to real-world data sources via Public RSS Feeds.
    Fetches live news and claims from fact-checking and news organizations.
    """
    def __init__(self):
        # We use a mix of reliable news and Fact-Checking sites which often
        # report on (and quote) disinformation claims circulating on social media.
        self.sources = [
            'http://feeds.bbci.co.uk/news/world/rss.xml',  # Reliable: BBC World
            'https://www.snopes.com/feed/',               # Fact-checker: Snopes
            'https://maldita.es/malditobulo/feed/'        # Fact-checker Spanish: Maldita.es
        ]

    def _clean_html(self, raw_html: str) -> str:
        """Removes HTML tags from RSS content to get plain text."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text(separator=" ").strip()

    def fetch_live_posts(self) -> list:
        """
        Fetches live posts from the defined RSS sources.
        Returns a list of dictionaries formatted for the DisinformationPipeline.
        """
        formatted_posts = []

        print("Fetching live data from real-world RSS feeds...")

        for url in self.sources:
            try:
                # feedparser automatically handles HTTP requests safely
                feed = feedparser.parse(url)

                if feed.entries:
                    print(f"  Successfully fetched {len(feed.entries[:5])} items from: {url}")
                else:
                    print(f"  Failed or no entries found for: {url}")

                # We limit to the top 5 most recent posts per feed to keep the demo quick
                for entry in feed.entries[:5]:

                    # Extract text content cleanly
                    content = entry.get('summary', '')
                    if 'content' in entry:
                        content = entry.content[0].value
                    clean_content = self._clean_html(content)

                    # Try to extract an image if available in standard RSS enclosures
                    media_url = ""
                    if 'media_content' in entry and len(entry.media_content) > 0:
                        media_url = entry.media_content[0].get('url', '')
                    elif 'links' in entry:
                        for link in entry.links:
                            if 'image' in link.get('type', ''):
                                media_url = link.get('href', '')
                                break

                    formatted_post = {
                        "headline": entry.get('title', ''),
                        "content": clean_content[:1000], # Keep first 1000 chars
                        "source_url": entry.get('link', ''),
                        "media_url": media_url,
                        "reported_date": datetime.now().strftime("%Y-%m-%d"), # Assume today if parsing fails

                        # Mock a user profile since RSS doesn't have social profiles
                        "user_profile": {
                            "created_at": "2015-01-01",
                            "followers_count": 100000,
                            "following_count": 50,
                            "has_profile_pic": True,
                            "statuses_count": 5000
                        }
                    }
                    formatted_posts.append(formatted_post)

            except Exception as e:
                print(f"  Error fetching from {url}: {e}")

        print(f"Total live posts fetched and formatted: {len(formatted_posts)}")
        return formatted_posts

# If run independently, just fetch and print one post to test
if __name__ == "__main__":
    ingestion = RealWorldDataIngestion()
    posts = ingestion.fetch_live_posts()
    if posts:
        print("\nSample Real Post:")
        print(json.dumps(posts[0], indent=2, ensure_ascii=False))
