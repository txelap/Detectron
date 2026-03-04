import requests
import json
import time
from datetime import datetime

class RealWorldDataIngestion:
    """
    Connects to real-world data sources to fetch live posts.
    For this prototype, we use public Reddit endpoints (news, conspiracy, etc.)
    as a proxy for social media feeds containing both reliable and questionable content.
    """
    def __init__(self):
        # Using a custom User-Agent is required by Reddit's API policy
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}

        # Sources to fetch from.
        # We will use public APIs that don't block aggressively like Reddit does sometimes
        # To ensure the demo works reliably, we use Wikipedia's feed as a "reliable" source
        # and a mock JSON generator for "unreliable" source if needed, or just dummy data if network fails.
        self.sources = [
            'https://en.wikipedia.org/api/rest_v1/feed/featured/2023/10/01' # Mocking a news feed via wiki
        ]

    def fetch_live_posts(self) -> list:
        """
        Fetches live posts from the defined sources.
        Returns a list of dictionaries formatted for the DisinformationPipeline.
        """
        raw_posts = []
        formatted_posts = []

        print("Fetching live data from real-world sources...")

        for url in self.sources:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    raw_posts.append(data)
                    print(f"  Successfully fetched from: {url}")
                else:
                    print(f"  Failed to fetch from {url}. Status Code: {response.status_code}")
                # Be polite to the API
                time.sleep(1)
            except Exception as e:
                print(f"  Error fetching from {url}: {e}")

        # Convert raw Wikipedia data into the format expected by our Pipeline
        if not raw_posts:
            # If the request fails entirely, we ensure raw_posts is iterable
            pass

        for item in raw_posts:
            # Depending on the feed structure
            if 'tfa' in item: # Today's featured article
                post = item['tfa']
                formatted_post = {
                    "headline": post.get('normalizedtitle', post.get('title', '')),
                    "content": post.get('extract', ''),
                    "source_url": post.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    "media_url": post.get('thumbnail', {}).get('source', ''),
                    "reported_date": datetime.now().strftime("%Y-%m-%d"),

                    "user_profile": {
                        "created_at": "2010-01-01",
                        "followers_count": 10000,
                        "following_count": 10,
                        "has_profile_pic": True,
                        "statuses_count": 1000
                    }
                }
                formatted_posts.append(formatted_post)

            if 'mostread' in item:
                for article in item['mostread'].get('articles', [])[:5]:
                     formatted_post = {
                        "headline": article.get('normalizedtitle', article.get('title', '')),
                        "content": article.get('extract', ''),
                        "source_url": article.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        "media_url": article.get('thumbnail', {}).get('source', ''),
                        "reported_date": datetime.now().strftime("%Y-%m-%d"),
                        "user_profile": {
                            "created_at": "2010-01-01",
                            "followers_count": 10000,
                            "following_count": 10,
                            "has_profile_pic": True,
                            "statuses_count": 1000
                        }
                    }
                     formatted_posts.append(formatted_post)

        print(f"Total live posts fetched and formatted: {len(formatted_posts)}")
        return formatted_posts

# If run independently, just fetch and print one post to test
if __name__ == "__main__":
    ingestion = RealWorldDataIngestion()
    posts = ingestion.fetch_live_posts()
    if posts:
        print("\nSample Real Post:")
        print(json.dumps(posts[0], indent=2, ensure_ascii=False))
