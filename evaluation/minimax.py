import os, chess, bot

QUIESCENCE_SEARCH_DEPTH: int = 5
SEARCH_DEPTH = 2

piece_values = [0, 1, 3, 3, 5, 9, 200]
current_board: chess.Board = None

def minimax(board: chess.Board, depth: int) -> tuple[float, chess.Move]:
    return _minimax(board, depth, float('-inf'), float('inf')) 
MVV_LVA = [    
    [0,  0,  0,  0,  0,  0,  0,  0],   
    [0, 15, 14, 13, 12, 11, 10,  0], 
    [0, 25, 24, 23, 22, 21, 20,  0],
    [0, 35, 34, 33, 32, 31, 30,  0], 
    [0, 45, 44, 43, 42, 41, 40,  0], 
    [0, 55, 54, 53, 52, 51, 50,  0]
]
def sort_list(move: chess.Move) -> int:
    if current_board.is_capture(move):
        if current_board.is_en_passant(move):
            return 15
        else:
            return MVV_LVA[current_board.piece_type_at(move.to_square)][current_board.piece_type_at(move.from_square)]
    return 0

def _minimax(board: chess.Board, depth: int, alpha: float, beta: float) -> tuple[float, chess.Move]:
    global current_board
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return float('-inf'), None
        else:
            return float('inf'), None

    if board.is_stalemate():
        return 0, None
    
    if depth < 1:
        return quiescence_search(board, alpha, beta, 1), None
        # return model_eval(board), None
    
    best_move = None

    legal_moves = list(board.legal_moves)
    current_board = board
    legal_moves.sort(key=sort_list, reverse=True)

    if board.turn:
        maximum = float('-inf')
        for move in legal_moves:
            board.push(move)
            value = _minimax(board, depth - 1, alpha, beta)[0]
            board.pop()
            if value > maximum:
                maximum = value
                best_move = move
            if maximum > alpha:
                alpha = maximum
            if beta <= alpha:
                break
        return maximum, best_move
    else:
        minimum = float('inf')
        for move in legal_moves:
            board.push(move)
            value = _minimax(board, depth - 1, alpha, beta)[0]
            board.pop()
            if value < minimum:
                minimum = value
                best_move = move
            if minimum < beta:
                beta = minimum
            if beta <= alpha:
                break
        return minimum, best_move
   
def quiescence_search(board: chess.Board, alpha: float, beta: float, currentDepth: int) -> float:
    
    if currentDepth == QUIESCENCE_SEARCH_DEPTH:
        return model_eval(board)

    favorable_moves = []
    moves = board.legal_moves

    for move in moves:
        if is_favorable_move(board, move):
            favorable_moves.append(move)

    if favorable_moves == []:
        return model_eval(board)
    
    
    if board.turn:
        maximum = float('-inf')
        for move in favorable_moves:
            board.push(move)
            value = quiescence_search(board, alpha, beta, currentDepth + 1)
            board.pop()
            if value > maximum:
                maximum = value
            if maximum > alpha:
                alpha = maximum
            if beta <= alpha:
                break
        return maximum
    else:
        minimum = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = quiescence_search(board, alpha, beta, currentDepth + 1)
            board.pop()
            if value < minimum:
                minimum = value
            if minimum < beta:
                beta = minimum
            if beta <= alpha:
                break
        return minimum

def is_favorable_move(board: chess.Board, move: chess.Move) -> bool:
    if move.promotion is not None:
        return True
    if board.is_capture(move) and not board.is_en_passant(move):
        if piece_values[board.piece_type_at(move.from_square)] < piece_values[board.piece_type_at(move.to_square)] \
         or len(board.attackers(board.turn, move.to_square)) > len(board.attackers(not board.turn, move.to_square)):
            return True
    return False

def model_eval(board: chess.Board) -> float:
    return bot.evaluate(board)

def who(player):
    return "White" if player == chess.WHITE else "Black"


def human_player(board: chess.Board) -> chess.Move:
    print(board)
    san = input(f"{who(board.turn)}'s move:")
    legal_san_moves = [board.san(move) for move in board.legal_moves]

    while san not in legal_san_moves:
        print(f"Legal moves: {(', '.join(sorted(legal_san_moves)))}")
        san = input(f"{who(board.turn)}'s move:")

    return board.parse_san(san)

def play_game(human_color: int = chess.WHITE):
    board = chess.Board()
    
    while not board.is_game_over(claim_draw=True):
        if board.turn == human_color:
            move = human_player(board)
        else:
            ev, move = minimax(board, SEARCH_DEPTH)   

        os.system('cls')
        if not board.turn == human_color:
            print(f'Evaluation: {ev}')
            print(f'Bot Move: {board.san(move)}')
    
        board.push(move) 

    result = None
    if board.is_checkmate():
        msg = "checkmate: " + who(not board.turn) + " wins!"
        result = not board.turn
    elif board.is_stalemate():
        msg = "draw: stalemate"
    elif board.is_fivefold_repetition():
        msg = "draw: fivefold repetition"
    elif board.is_insufficient_material():
        msg = "draw: insufficient material"
    elif board.can_claim_draw():
        msg = "draw: claim"
    
    print(msg)
    return result, msg, board

play_game()
