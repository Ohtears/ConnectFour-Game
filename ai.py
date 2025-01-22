import math
import numpy as np

from utils import ROW_COUNT, COLUMN_COUNT, is_valid_location, get_next_open_row, winning_move, drop_piece

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, 2):
                return (None, 100000000000000)
            elif winning_move(board, 1):
                return (None, -10000000000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, 2))
    if maximizingPlayer:
        value = -math.inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 2)
            new_score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            elif new_score == value:
                # Break ties by favoring the center column or specific criteria
                center_column = COLUMN_COUNT // 2
                if abs(col - center_column) < abs(column - center_column):
                    column = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break
        print(f"Evaluating column {col} for maximizing player: Score = {new_score}")
        return column, value
    else:
        value = math.inf
        column = np.random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 1)
            new_score = minimax(temp_board, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
    print(f"Evaluating column {col} for minimizing player: Score = {new_score}")
    return column, value

def ai_move(board, depth):
    column, score = minimax(board, depth, -math.inf, math.inf, True)
    print(f"AI's final decision: Move at column {column} with score {score}")
    return column

def get_valid_locations(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]

def is_terminal_node(board):
    return winning_move(board, 1) or winning_move(board, 2) or len(get_valid_locations(board)) == 0

def score_position(board, piece):
    score = 0
    opp_piece = 1 if piece == 2 else 2

    # Center column preference
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 4  # Increase weight for center

    # Horizontal scoring
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + 4]
            score += evaluate_window(window, piece)

    # Vertical scoring
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + 4]
            score += evaluate_window(window, piece)

    # Positive diagonal scoring
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Negative diagonal scoring
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Penalize opponent's potential wins
    if winning_move(board, opp_piece):
        score -= 10000

    return score

def evaluate_window(window, piece):
    score = 0
    opp_piece = 1 if piece == 2 else 2

    if window.count(piece) == 4:  # Winning move
        score += 1000
    elif window.count(piece) == 3 and window.count(0) == 1:  # 3-in-a-row setup
        score += 100
    elif window.count(piece) == 2 and window.count(0) == 2:  # Potential setup
        score += 10

    if window.count(opp_piece) == 3 and window.count(0) == 1:  # Block opponent
        score -= 150  # Increase penalty to prioritize defense

    # Reward double threats (two separate winning opportunities)
    if window.count(piece) == 2 and window.count(opp_piece) == 0:
        score += 50

    return score

