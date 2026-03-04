import numpy as np
import tensorflow as tf
from nn_models import (
    create_nlp_model,
    create_tabular_model,
    create_vision_model,
    create_meta_ensemble_model
)

def generate_dummy_data(num_samples=1000):
    """
    Generates dummy data to simulate the output of preprocessing steps.
    """
    print(f"Generating {num_samples} dummy samples for training...")

    # NLP Data: Tokenized sequences (e.g., max_length=200)
    vocab_size = 10000
    X_text = np.random.randint(0, vocab_size, size=(num_samples, 200))

    # Tabular Data: Metadata features (e.g., 5 numerical/categorical features)
    X_meta = np.random.rand(num_samples, 5)

    # Vision Data: Image arrays (e.g., 128x128 RGB images)
    X_image = np.random.rand(num_samples, 128, 128, 3)

    # Labels: 0 for trustworthy, 1 for disinformation
    # Making it slightly imbalanced to simulate real world
    y = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])

    return X_text, X_meta, X_image, y

def main():
    print("--- Initializing Neural Network Ensemble ---")

    # 1. Instantiate Sub-Models
    nlp_model = create_nlp_model()
    tabular_model = create_tabular_model()
    vision_model = create_vision_model()

    # 2. Instantiate Meta-Ensemble
    ensemble = create_meta_ensemble_model(nlp_model, tabular_model, vision_model)

    # Display the architecture summary
    ensemble.summary()

    # 3. Generate Dummy Training Data
    X_text, X_meta, X_image, y_true = generate_dummy_data(num_samples=500)

    print("\n--- Starting Training Process ---")
    # 4. Train the Ensemble Model
    # We pass a list of inputs matching the order defined in the ensemble model
    history = ensemble.fit(
        x=[X_text, X_meta, X_image],
        y=y_true,
        epochs=3,          # Keep it short for demonstration
        batch_size=32,
        validation_split=0.2, # Use 20% of data for validation
        verbose=1
    )

    print("\n--- Training Complete ---")
    print("Final Validation Accuracy: {:.2f}%".format(history.history['val_accuracy'][-1] * 100))

    # Optional: Save the model weights
    # ensemble.save('disinfo_ensemble_weights.h5')
    print("Model ready for inference!")

if __name__ == "__main__":
    main()
