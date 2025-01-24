import math
import numpy as np

from utils import ROW_COUNT, COLUMN_COUNT, is_valid_location, get_next_open_row, winning_move, drop_piece

def minimax(board, depth, alpha, beta, maximizingPlayer):
    is_terminal = is_terminal_node(board)
    valid_locations = get_valid_locations(board)
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, 2):  
                return None, 10e10
            elif winning_move(board, 1):  
                return None, -10e10
            else: 
                return None, 0
        else:  
            return None, score_position(board, 2)

    if maximizingPlayer:
        value = -math.inf
        best_column = np.random.choice(valid_locations) 
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 2)  
            new_score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_column = col
            alpha = max(alpha, value)
            if alpha >= beta:  
                break
        return best_column, value

    else:
        value = math.inf
        best_column = np.random.choice(valid_locations)  
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, 1)  
            new_score = minimax(temp_board, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_column = col
            beta = min(beta, value)
            if alpha >= beta: 
                break
        return best_column, value

def AI_move(board, depth):
    column, score = minimax(board, depth, -math.inf, math.inf, True)
    # print(f"AI's final decision: Move at column {column} with score {score}")
    return column

def get_valid_locations(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]


def is_terminal_node(board):
    valid_locations = get_valid_locations(board)
    player_win = winning_move(board, 1)
    ai_win = winning_move(board, 2)
    is_draw = len(valid_locations) == 0
    if player_win or ai_win or is_draw:
        pass
        # print(f"Terminal node detected: Player win={player_win}, AI win={ai_win}, Draw={is_draw}")
    return player_win or ai_win or is_draw


def score_position(board, piece):
    score = 0
    opp_piece = 1 if piece == 2 else 2

    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 6 # center column

    left_center_array = [int(i) for i in list(board[:, (COLUMN_COUNT // 2) - 1])]
    right_center_array = [int(i) for i in list(board[:, (COLUMN_COUNT // 2) + 1])]
    left_center_count = left_center_array.count(piece)
    right_center_count = right_center_array.count(piece)
    score += (left_center_count + right_center_count) * 3  # adjacent center

    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + 4]
            score += evaluate_window(window, piece)

    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + 4]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

            window = [board[r + 3 - i][c + i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score


def is_valid_window(board, row, col, direction):
    if direction == 'horizontal':
        return all(board[row][col + i] != 0 or row == 0 or board[row - 1][col + i] != 0 for i in range(4))
    elif direction == 'vertical':
        return row == 0 or all(board[row - 1][col] != 0 for i in range(4))
    elif direction == 'positive_diagonal':
        return all(board[row + i][col + i] != 0 or row + i == 0 or board[row + i - 1][col + i] != 0 for i in range(4))
    elif direction == 'negative_diagonal':
        return all(board[row + 3 - i][col + i] != 0 or row + 3 - i == 0 or board[row + 3 - i - 1][col + i] != 0 for i in range(4))
    return False

def evaluate_window(window, piece):
    score = 0
    opp_piece = 1 if piece == 2 else 2

    if window.count(piece) == 4:  # win
        score += 800
    elif window.count(piece) == 3 and window.count(0) == 1:  # creating 3-row
        score += 100
    elif window.count(piece) == 2 and window.count(0) == 2:  # creating 2-row
        score += 10

    if window.count(opp_piece) == 3 and window.count(0) == 1:  # block
        score -= 150

    return score