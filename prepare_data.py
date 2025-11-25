import chess
import chess.pgn
import numpy as np

# Mapeo de piezas a índices para los canales del tensor
PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def board_to_tensor(board):
    """
    Convierte un objeto chess.Board a un tensor de 8x8x13.
    - 12 canales para las piezas (6 para blancas, 6 para negras).
    - 1 canal para indicar el turno (1s si es el turno de las blancas, 0s si es el de las negras).
    """
    tensor = np.zeros((8, 8, 13), dtype=np.uint8)

    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                # El canal se determina por el tipo de pieza y su color
                # Canales 0-5 son para piezas blancas, 6-11 para negras
                channel_index = PIECE_TO_INDEX[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel_index += 6
                tensor[rank, file, channel_index] = 1

    # El canal 12 indica de quién es el turno
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1
    else:
        tensor[:, :, 12] = 0

    return tensor

def process_pgn_file(pgn_path):
    """
    Procesa un archivo PGN y extrae pares (posición, siguiente_movimiento).
    """
    board_positions = []
    next_moves = []

    with open(pgn_path) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():
                # Guardamos la posición ANTES de hacer el movimiento
                board_tensor = board_to_tensor(board)

                # El movimiento se guarda en formato UCI (e.g., 'e2e4')
                move_uci = move.uci()

                board_positions.append(board_tensor)
                next_moves.append(move_uci)

                # Hacemos el movimiento en el tablero para la siguiente iteración
                board.push(move)

    return np.array(board_positions), np.array(next_moves)

if __name__ == "__main__":
    X, y = process_pgn_file('games.pgn')

    print(f"Procesadas {len(X)} posiciones.")
    print("Forma de los datos de entrada (X):", X.shape)
    print("Forma de las etiquetas (y):", y.shape)

    # Guardamos los datos procesados para usarlos en el entrenamiento
    np.savez('processed_data.npz', X=X, y=y)
    print("Datos guardados en 'processed_data.npz'")
