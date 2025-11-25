import tensorflow as tf
from tensorflow.keras import layers, models

def build_chess_model(num_possible_moves):
    """
    Construye un modelo de red neuronal convolucional para ajedrez.

    Args:
        num_possible_moves (int): El número total de movimientos posibles en ajedrez,
                                  que será el tamaño de la capa de salida.

    Returns:
        Un modelo de Keras compilado.
    """
    # La entrada tiene la forma de nuestro tensor: un tablero de 8x8 con 13 canales
    input_shape = (8, 8, 13)

    model = models.Sequential([
        # Capa convolucional para aprender patrones espaciales
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Segunda capa convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Aplanamos la salida para pasarla a las capas densas
        layers.Flatten(),

        # Capa densa para aprender combinaciones de características
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        # La capa de salida tiene una neurona por cada posible movimiento.
        # La activación 'softmax' nos da una distribución de probabilidad sobre los movimientos.
        layers.Dense(num_possible_moves, activation='softmax')
    ])

    # Compilamos el modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # Este es un número de ejemplo. En el script de entrenamiento, lo calcularemos
    # basándonos en los movimientos únicos de nuestro conjunto de datos.
    NUM_MOVES_EXAMPLE = 1800

    chess_nn = build_chess_model(NUM_MOVES_EXAMPLE)

    # Imprimimos un resumen de la arquitectura del modelo
    chess_nn.summary()
