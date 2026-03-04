import json
from datetime import datetime
import feedparser
from bs4 import BeautifulSoup
import time
from mastodon import Mastodon

class RealWorldDataIngestion:
    """
    Connects to real-world data sources via Public RSS Feeds and Social Media APIs (Mastodon).
    Fetches live news, claims, and social posts.
    """
    def __init__(self):
        # RSS Feeds
        self.rss_sources = [
            'http://feeds.bbci.co.uk/news/world/rss.xml',  # Reliable: BBC World
            'https://www.snopes.com/feed/',               # Fact-checker: Snopes
            'https://maldita.es/malditobulo/feed/'        # Fact-checker Spanish: Maldita.es
        ]

        # Mastodon setup (unauthenticated read-only access)
        self.mastodon_instance = 'https://mastodon.social'

    def _clean_html(self, raw_html: str) -> str:
        """Removes HTML tags from content to get plain text."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text(separator=" ").strip()

    def fetch_rss_posts(self) -> list:
        """Fetches live posts from the defined RSS sources."""
        posts = []
        print("Fetching live data from real-world RSS feeds...")

        for url in self.rss_sources:
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
                    posts.append(formatted_post)

            except Exception as e:
                import traceback
                print(f"  Error fetching from {url}: {e}")
                traceback.print_exc()

        return posts

    def fetch_mastodon_posts(self, limit=5) -> list:
        """
        Fetches the public local timeline from a Mastodon instance.
        This provides real "social media" unstructured text and profiles.
        """
        posts = []
        print(f"Fetching live data from social media: {self.mastodon_instance}...")
        try:
            m = Mastodon(api_base_url=self.mastodon_instance)
            # Fetch the local public timeline
            toots = m.timeline_local(limit=limit)
            print(f"  Successfully fetched {len(toots)} items from: Mastodon")

            for toot in toots:
                content = self._clean_html(toot.get('content', ''))
                if not content:
                    continue

                # Get media if available
                media_url = ""
                if toot.get('media_attachments'):
                    for attachment in toot['media_attachments']:
                        if attachment.get('type') == 'image':
                            media_url = attachment.get('url', '')
                            break

                # Parse account info
                account = toot.get('account', {})
                created_at = account.get('created_at')
                # Mastodon dates are often datetime objects or ISO strings
                if isinstance(created_at, datetime):
                    created_date_str = created_at.strftime("%Y-%m-%d")
                else:
                    created_date_str = "2020-01-01"

                formatted_post = {
                    "headline": "Social Media Post", # Mastodon usually doesn't have headlines
                    "content": content,
                    "source_url": toot.get('url', self.mastodon_instance),
                    "media_url": media_url,
                    "reported_date": datetime.now().strftime("%Y-%m-%d"),

                    "user_profile": {
                        "created_at": created_date_str,
                        "followers_count": account.get('followers_count', 0),
                        "following_count": account.get('following_count', 0),
                        "has_profile_pic": 'missing.png' not in account.get('avatar', ''),
                        "statuses_count": account.get('statuses_count', 0)
                    }
                }
                posts.append(formatted_post)
        except Exception as e:
             print(f"  Error fetching from Mastodon: {e}")

        return posts

    def fetch_live_posts(self) -> list:
        """
        Combines posts from all sources (RSS and Mastodon) and returns a unified list.
        """
        all_posts = []
        all_posts.extend(self.fetch_rss_posts())
        all_posts.extend(self.fetch_mastodon_posts(limit=10)) # Get 10 live toots

        print(f"\nTotal combined live posts fetched and formatted: {len(all_posts)}")
        return all_posts

# If run independently, just fetch and print one post to test
if __name__ == "__main__":
    ingestion = RealWorldDataIngestion()
    posts = ingestion.fetch_live_posts()
    if posts:
        print("\nSample Real Post:")
        print(json.dumps(posts[0], indent=2, ensure_ascii=False))
