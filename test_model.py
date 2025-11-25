import chess
import numpy as np
import tensorflow as tf
from prepare_data import board_to_tensor
import json

# --- Cargar Modelo y Mapeo ---
model = tf.keras.models.load_model('chess_model.h5')
with open('move_mapping.json', 'r') as f:
    move_mapping = json.load(f)
    # Las claves en JSON se guardan como strings, las convertimos de nuevo a enteros
    move_mapping = {int(k): v for k, v in move_mapping.items()}

print("Modelo y mapeo de movimientos cargados.")

# --- Preparar una Posición de Prueba ---
# Empezaremos con el tablero en su posición inicial.
board = chess.Board()
print("\nTablero inicial:")
print(board)

# Convertimos el tablero al formato de tensor que el modelo espera.
board_tensor = board_to_tensor(board)
# El modelo espera un 'batch' de datos, así que añadimos una dimensión extra.
board_tensor = np.expand_dims(board_tensor, axis=0)

# --- Realizar la Predicción ---
predictions = model.predict(board_tensor)

# La predicción es un array de probabilidades. Encontramos el índice del movimiento
# con la probabilidad más alta.
predicted_move_index = np.argmax(predictions[0])

# --- Decodificar y Mostrar el Resultado ---
predicted_move_uci = move_mapping.get(predicted_move_index, "Movimiento desconocido")

print(f"\nLa red neuronal predice que el mejor movimiento es: {predicted_move_uci}")

# --- Verificación Opcional ---
# ¿Es un movimiento legal?
try:
    move = chess.Move.from_uci(predicted_move_uci)
    if move in board.legal_moves:
        print("El movimiento predicho es legal.")
    else:
        print("¡Atención! El movimiento predicho NO es legal.")
except:
    print("El movimiento predicho no es un formato UCI válido.")
