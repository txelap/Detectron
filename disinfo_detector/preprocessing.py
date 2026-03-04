import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import requests
from PIL import Image
from io import BytesIO

class DataPreprocessor:
    """
    Converts real-world data (text, URLs, metadata dictionaries)
    into numerical Tensors that the Neural Network can digest.
    """
    def __init__(self, vocab_size=10000, max_sequence_length=200):
        # NLP Preprocessor: Learns the vocabulary and tokenizes text
        self.vectorizer = TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=max_sequence_length
        )
        self.is_vectorizer_adapted = False

    def adapt_vectorizer(self, corpus: list):
        """
        Adapts the TextVectorization layer to the vocabulary of a given corpus.
        Must be called before processing text.
        """
        print(f"Adapting NLP vectorizer to {len(corpus)} text samples...")
        self.vectorizer.adapt(corpus)
        self.is_vectorizer_adapted = True
        print("Vocabulary learned.")

    def process_text(self, text: str) -> np.ndarray:
        """Converts a string of text into a sequence of integer tokens."""
        if not self.is_vectorizer_adapted:
            # Fallback for inference before full training
            # In a real app, you load a pre-adapted vocabulary
            return np.zeros((200,), dtype=np.int32)

        # The vectorizer expects a batch, so we wrap it in a list and extract the first element
        return self.vectorizer([text]).numpy()[0]

    def process_metadata(self, profile: dict, source_score: float) -> np.ndarray:
        """
        Converts user metadata and heuristic pipeline scores into a fixed-size numerical vector.
        Features (5 total):
        0: followers count (normalized roughly)
        1: following ratio
        2: has profile pic (0 or 1)
        3: statuses count (normalized roughly)
        4: heuristic source unreliability score (from pipeline)
        """
        # Feature 0
        f_count = profile.get("followers_count", 0)
        norm_followers = min(f_count / 10000.0, 1.0)

        # Feature 1
        followers = max(f_count, 1)
        following = profile.get("following_count", 0)
        ratio = min(following / followers, 10.0) / 10.0 # Normalize 0-1

        # Feature 2
        has_pic = 1.0 if profile.get("has_profile_pic", True) else 0.0

        # Feature 3
        statuses = profile.get("statuses_count", 0)
        norm_statuses = min(statuses / 50000.0, 1.0)

        # Feature 4
        source_unreliability = source_score

        return np.array([norm_followers, ratio, has_pic, norm_statuses, source_unreliability], dtype=np.float32)

    def process_image(self, image_url: str, target_size=(128, 128)) -> np.ndarray:
        """
        Downloads an image from a URL, resizes it, and converts it to a normalized numpy array.
        Returns a blank image array if the URL is invalid or download fails.
        """
        # Blank/Black image fallback
        blank_image = np.zeros((*target_size, 3), dtype=np.float32)

        if not image_url or not image_url.startswith('http'):
            return blank_image

        try:
            # Fetch the image
            response = requests.get(image_url, timeout=5)
            if response.status_code == 200:
                # Open with Pillow
                img = Image.open(BytesIO(response.content))
                # Ensure RGB (removes alpha channels etc)
                img = img.convert('RGB')
                # Resize to target shape
                img = img.resize(target_size)
                # Convert to numpy array and normalize pixel values between 0 and 1
                img_array = np.array(img, dtype=np.float32) / 255.0
                return img_array
        except Exception as e:
            # Silently catch timeouts or bad images to keep the pipeline moving
            pass

        return blank_image
