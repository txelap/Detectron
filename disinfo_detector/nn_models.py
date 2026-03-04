import tensorflow as tf
from tensorflow.keras import layers, Model

def create_nlp_model(vocab_size=10000, embedding_dim=128, max_length=200):
    """
    Sub-network 1: NLP Model
    Used for: Clickbait detection, Emotion analysis, Claim extraction.
    Takes tokenized text sequences and returns a feature representation.
    """
    input_text = layers.Input(shape=(max_length,), name="text_input")
    x = layers.Embedding(vocab_size, embedding_dim)(input_text)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    # Output is a dense feature vector representing the text
    output = layers.Dense(32, activation='relu', name="nlp_features")(x)

    return Model(inputs=input_text, outputs=output, name="NLP_Model")

def create_tabular_model(num_features=5):
    """
    Sub-network 2: Tabular Metadata Model
    Used for: Profile analysis (bot detection), Source credibility features, Date deltas.
    Takes numerical/categorical metadata and returns a feature representation.
    """
    input_meta = layers.Input(shape=(num_features,), name="meta_input")
    x = layers.Dense(32, activation='relu')(input_meta)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    # Output is a dense feature vector representing metadata
    output = layers.Dense(8, activation='relu', name="meta_features")(x)

    return Model(inputs=input_meta, outputs=output, name="Tabular_Model")

def create_vision_model(image_shape=(128, 128, 3)):
    """
    Sub-network 3: Vision Model (CNN)
    Used for: Deepfake detection, AI image artifact detection.
    Takes image arrays and returns a feature representation.
    """
    input_img = layers.Input(shape=image_shape, name="image_input")
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    # Output is a dense feature vector representing the image
    output = layers.Dense(16, activation='relu', name="vision_features")(x)

    return Model(inputs=input_img, outputs=output, name="Vision_Model")

def create_meta_ensemble_model(nlp_model, tabular_model, vision_model):
    """
    The 'Network of Networks': Meta-Ensemble
    Combines the feature outputs of the sub-networks to make a final prediction.
    """
    # 1. Get inputs from sub-models
    text_input = nlp_model.input
    meta_input = tabular_model.input
    image_input = vision_model.input

    # 2. Get feature outputs from sub-models
    nlp_features = nlp_model.output
    meta_features = tabular_model.output
    vision_features = vision_model.output

    # 3. Concatenate all features
    combined = layers.Concatenate()([nlp_features, meta_features, vision_features])

    # 4. Final Meta-Classifier
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)

    # Final output: Probability of being Disinformation (0 to 1)
    final_output = layers.Dense(1, activation='sigmoid', name="disinfo_probability")(x)

    # Create the unified model
    ensemble_model = Model(
        inputs=[text_input, meta_input, image_input],
        outputs=final_output,
        name="Disinformation_Ensemble_Network"
    )

    ensemble_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return ensemble_model
