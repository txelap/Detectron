import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from create_model import build_chess_model
import json

# --- Cargar Datos ---
with np.load('processed_data.npz') as data:
    X = data['X']
    y = data['y']

print(f"Datos cargados. X shape: {X.shape}, y shape: {y.shape}")

# --- Preparar Etiquetas (Movimientos) ---

# 1. Crear un mapeo de cada movimiento único a un entero
# Esto es necesario porque la red neuronal trabaja con números.
label_encoder = LabelEncoder()
integer_encoded_moves = label_encoder.fit_transform(y)
print(f"Se encontraron {len(label_encoder.classes_)} movimientos únicos.")

# Guardamos el mapeo para poder decodificar las predicciones más tarde
move_mapping = {i: move for i, move in enumerate(label_encoder.classes_)}
with open('move_mapping.json', 'w') as f:
    json.dump(move_mapping, f)

# 2. Convertir los enteros a un formato 'one-hot'
# La red predice probabilidades, por lo que necesitamos un vector donde
# solo el movimiento correcto tenga un 1 y el resto sean 0.
onehot_encoder = OneHotEncoder(sparse_output=False)
y_categorical = onehot_encoder.fit_transform(integer_encoded_moves.reshape(-1, 1))

# --- Construir y Entrenar el Modelo ---

# El número de neuronas de salida debe coincidir con el número de movimientos únicos
num_possible_moves = len(label_encoder.classes_)
model = build_chess_model(num_possible_moves)

print("\nIniciando entrenamiento del modelo...")
# Entrenamos el modelo con nuestros datos.
# Con un conjunto de datos tan pequeño, unas pocas épocas son suficientes.
model.fit(X, y_categorical, epochs=15, batch_size=2, validation_split=0.1)

# --- Guardar el Modelo Entrenado ---
model.save('chess_model.h5')
print("\nModelo entrenado y guardado como 'chess_model.h5'")
