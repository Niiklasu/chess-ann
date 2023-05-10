import os
import sys
import time
import uuid
import bot
import chess
import chess.engine
import chess.polyglot
import chess.svg

from collections import defaultdict
from collections import OrderedDict

QUIESCENCE_SEARCH_DEPTH: int = 20
TABLE_SIZE: int = 1.84e19
TIMEOUT_SECONDS: int = 10



best_move = None
current_depth = 0
global_best_move = None
is_timeout = False
move_scores = defaultdict(dict)
piece_zobrist_values = []
repetition_table = {}
start_time = 0.0
transposition_table = None
zobrist_turn = 0

piece_values = {
    chess.BISHOP: 330,
    chess.KING: 20_000,
    chess.KNIGHT: 320,
    chess.PAWN: 100,
    chess.QUEEN: 900,
    chess.ROOK: 500,
}

zobrist_values_white = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}
zobrist_values_black = {
    chess.PAWN: 7,
    chess.KNIGHT: 8,
    chess.BISHOP: 9,
    chess.ROOK: 10,
    chess.QUEEN: 11,
    chess.KING: 12,
}

def init_zobrist_list():
    global piece_zobrist_values
    global zobrist_turn

    zobrist_turn = uuid.uuid4().int & (1 << 64) - 1
    for i in range(0, 64):
        NO_PIECE = uuid.uuid4().int & (1 << 64) - 1
        WHITE_PAWN = uuid.uuid4().int & (1 << 64) - 1
        WHITE_KNIGHT = uuid.uuid4().int & (1 << 64) - 1
        WHITE_BISHOP = uuid.uuid4().int & (1 << 64) - 1
        WHITE_ROOK = uuid.uuid4().int & (1 << 64) - 1
        WHITE_QUEEN = uuid.uuid4().int & (1 << 64) - 1
        WHITE_KING = uuid.uuid4().int & (1 << 64) - 1
        BLACK_PAWN = uuid.uuid4().int & (1 << 64) - 1
        BLACK_KNIGHT = uuid.uuid4().int & (1 << 64) - 1
        BLACK_BISHOP = uuid.uuid4().int & (1 << 64) - 1
        BLACK_ROOK = uuid.uuid4().int & (1 << 64) - 1
        BLACK_QUEEN = uuid.uuid4().int & (1 << 64) - 1
        BLACK_KING = uuid.uuid4().int & (1 << 64) - 1

        piece_zobrist_values.append(
            [
                NO_PIECE,
                WHITE_PAWN,
                WHITE_KNIGHT,
                WHITE_BISHOP,
                WHITE_ROOK,
                WHITE_QUEEN,
                WHITE_KING,
                BLACK_PAWN,
                BLACK_KNIGHT,
                BLACK_BISHOP,
                BLACK_ROOK,
                BLACK_QUEEN,
                BLACK_KING,
            ]
        )


init_zobrist_list()

def zobrist_hash(board: chess.Board) -> int:
    global zobrist_turn

    zobrist_hash = 0
    if board.turn == chess.WHITE:
        zobrist_hash = zobrist_hash ^ zobrist_turn

    for square in range(64):
        piece = board.piece_at(square)
        if not piece:
            index = 0
        elif piece.color == chess.WHITE:
            index = zobrist_values_white.get(piece.piece_type)
        elif piece.color == chess.BLACK:
            index = zobrist_values_black.get(piece.piece_type)
        zobrist_hash = zobrist_hash ^ piece_zobrist_values[square][index]

    return zobrist_hash

def zobrist_move(board,move,zobrist_hash):
    white_to_move = board.turn
    from_square = move.from_square
    to_square = move.to_square
    moving_piece = board.piece_at(from_square)
    captured_piece = board.piece_at(to_square)

    if white_to_move:
        zobrist_hash = zobrist_hash ^ piece_zobrist_values[from_square][zobrist_values_white.get(moving_piece.piece_type)]
        zobrist_hash = zobrist_hash ^ piece_zobrist_values[from_square][0]
        if not captured_piece:
            zobrist_hash = zobrist_hash ^ piece_zobrist_values[to_square][0]
            zobrist_hash = zobrist_hash ^ piece_zobrist_values[to_square][zobrist_values_white.get(moving_piece.piece_type)]
        else:
            zobrist_hash = zobrist_hash ^ piece_zobrist_values[to_square][zobrist_values_black.get(captured_piece.piece_type)]
            zobrist_hash = zobrist_hash ^ piece_zobrist_values[to_square][zobrist_values_white.get(moving_piece.piece_type)]
    else:
        zobrist_hash = zobrist_hash ^ piece_zobrist_values[from_square][zobrist_values_black.get(moving_piece.piece_type)]
        zobrist_hash = zobrist_hash ^ piece_zobrist_values[from_square][0]
        if not captured_piece:
            zobrist_hash = zobrist_hash ^ piece_zobrist_values[to_square][0]
            zobrist_hash = zobrist_hash ^ piece_zobrist_values[to_square][zobrist_values_black.get(moving_piece.piece_type)]
        else:
            zobrist_hash = zobrist_hash ^ piece_zobrist_values[to_square][zobrist_values_white.get(captured_piece.piece_type)]
            zobrist_hash = zobrist_hash ^ piece_zobrist_values[to_square][zobrist_values_black.get(moving_piece.piece_type)]
            
    zobrist_hash = zobrist_hash^zobrist_turn
    return zobrist_hash

class LRUCache:
    def __init__(self, size):
        self.od = OrderedDict()
        self.size = size

    def get(self, key, default=None):
        try:
            self.od.move_to_end(key)
        except KeyError:
            return default
        return self.od[key]

    def __contains__(self, item):
        return item in self.od

    def __len__(self):
        return len(self.od)

    def __getitem__(self, key):
        self.od.move_to_end(key)
        return self.od[key]

    def __setitem__(self, key, value):
        try:
            del self.od[key]
        except KeyError:
            if len(self.od) == self.size:
                self.od.popitem(last=False)
        self.od[key] = value


transposition_table = LRUCache(TABLE_SIZE)

def iterative_deepening(board: chess.Board, depth: int, color: bool):
    global best_move
    global current_depth
    global global_best_move
    global is_timeout
    global start_time

    zobrist = zobrist_hash(board)
    increment_repetiton_table(zobrist)
    
    if board.legal_moves == 1:
        return board.legal_moves[0]

    is_timeout = False
    start_time = time.time()
    d = 0
    current_score = 0

    while True:
        if d > 1:
            global_best_move = best_move
            print(f"Completed search with depth {current_depth}. Best move so far: {global_best_move} (Score: {current_score})")
        if current_score == sys.maxsize or current_score == -sys.maxsize:
            return global_best_move
        current_depth = depth + d
        if color == chess.BLACK:
            current_score = minimize(board, current_depth, -sys.maxsize, sys.maxsize, color, zobrist)
        else:
            current_score = maximize(board, current_depth, -sys.maxsize, sys.maxsize, color, zobrist)
        d += 1
        if is_timeout:
            return global_best_move

def check_triple_repetition(zobrist_hash):
    global repetition_table
    if  zobrist_hash in repetition_table:
        times_encountered = repetition_table[zobrist_hash]
        if times_encountered > 2:
            return True
        else:
            return False
    else:
        return False

def increment_repetiton_table(zobrist_hash: int):
    if  zobrist_hash in repetition_table:
        times_encountered = repetition_table[zobrist_hash]
        times_encountered = times_encountered + 1
        repetition_table[zobrist_hash] = times_encountered
    else:
        repetition_table[zobrist_hash] = 1

def decrement_repetition_table(zobrist_hash: int):
    times_encountered = repetition_table[zobrist_hash]
    if times_encountered == 1:
        del repetition_table[zobrist_hash]
    else:
        times_encountered = times_encountered - 1
        repetition_table[zobrist_hash] = times_encountered

def maximize(board: chess.Board, depth: int, alpha: int, beta: int, color: int, zobrist: int) -> int:
    global is_timeout
    global start_time
    global best_move
    global global_best_move

    if time.time() - start_time > TIMEOUT_SECONDS:
        is_timeout = True
        return alpha
    
    
    if board.is_checkmate():
        return -sys.maxsize
    if board.is_stalemate():
        return 0
    if check_triple_repetition(zobrist):
        return 0

    if (zobrist, depth) in transposition_table:
        score, a, b = transposition_table[(zobrist, depth)]
        if a <= alpha and beta <= b:
            return score
        else:
            alpha = min(alpha, a)
            beta = max(beta, b)

    if depth < 1:
        return quiescence_search_maximize(board, alpha, beta, 1)

    score = alpha

    board_scores = move_scores.get(board.fen(), dict())
    moves = sorted(
        board.legal_moves, key=lambda move: -board_scores.get(move, 0),
    )

    for move in moves:
        new_zobrist = zobrist_move(board, move, zobrist)
        increment_repetiton_table(new_zobrist)
        board.push(move)
        move_score = minimize(board, depth - 1, score, beta, color, new_zobrist)
        move_scores[board.fen()][move] = move_score
        decrement_repetition_table(new_zobrist)
        board.pop()

        if move_score > score:
            score = move_score
            if depth == current_depth:
                best_move = move
            if score >= beta:
                break

    transposition_table[(zobrist, depth)] = score, alpha, beta
    return score

def minimize(board: chess.Board, depth: int, alpha: int, beta: int, color: int, zobrist) -> int:
    global best_move
    global global_best_move
    global is_timeout
    global start_time
    
    if time.time() - start_time > TIMEOUT_SECONDS:
        is_timeout = True
        return beta
    
    if board.is_checkmate():
        return sys.maxsize
    if board.is_stalemate():
        return 0
    if check_triple_repetition(zobrist):
        return 0
    
    if (zobrist, depth) in transposition_table:
        score, a, b = transposition_table[(zobrist, depth)]
        if a <= alpha and beta <= b:
            return score
        else:
            alpha = min(alpha, a)
            beta = max(beta , b)
    
    if depth < 1:
        return quiescence_search_minimize(board, alpha, beta, 1)
    
    score = beta
    
    board_scores = move_scores.get(board.fen(), dict())
    moves = sorted(
        board.legal_moves, key=lambda move: board_scores.get(move, 0),
    )

    for move in moves:
        new_zobrist = zobrist_move(board, move, zobrist)
        increment_repetiton_table(new_zobrist)
        board.push(move)
        move_score = maximize(board, depth - 1, alpha, score, color, new_zobrist)
        move_scores[board.fen()][move] = move_score
        decrement_repetition_table(new_zobrist)
        board.pop()

        if move_score < score:
            score = move_score
            if depth==current_depth:
                best_move = move
            if score <= alpha:
                break

    transposition_table[(zobrist, depth)] = score, alpha, beta
    return score

def eval_heuristic(board: chess.Board):
    return bot.evaluate(board)

def quiescence_search_maximize(board: chess.Board, alpha, beta, currentDepth: int):
    global best_move
    global global_best_move

    if currentDepth == QUIESCENCE_SEARCH_DEPTH:
        return eval_heuristic(board)

    favorable_moves = []
    moves = board.legal_moves

    for move in moves:
        if is_favorable_move(board, move):
            favorable_moves.append(move)

    if favorable_moves == []:
        return eval_heuristic(board)

    score = alpha
    for move in favorable_moves:
        board.push(move)
        move_score = quiescence_search_minimize(board, score, beta, currentDepth + 1)
        board.pop()
        if move_score > score:
            score = move_score
            if score >= beta:
                break

    return score

def quiescence_search_minimize(board: chess.Board, alpha, beta, currentDepth: int):
    global best_move
    global global_best_move

    if currentDepth == QUIESCENCE_SEARCH_DEPTH:
        return eval_heuristic(board)

    moves = board.legal_moves
    favorable_moves = []

    for move in moves:
        if is_favorable_move(board, move):
            favorable_moves.append(move)

    if favorable_moves == []:
        return eval_heuristic(board)

    score = beta
    for move in favorable_moves:
        board.push(move)
        move_score = quiescence_search_maximize(board, alpha, score, currentDepth + 1)
        board.pop()
        if move_score < score:
            score = move_score
            if score <= alpha:
                break

    return score

def is_favorable_move(board: chess.Board, move: chess.Move) -> bool:
    if move.promotion is not None:
        return True
    if board.is_capture(move) and not board.is_en_passant(move):
        if piece_values.get(board.piece_type_at(move.from_square)) < piece_values.get(
            board.piece_type_at(move.to_square)
        ) or len(board.attackers(board.turn, move.to_square)) > len(
            board.attackers(not board.turn, move.to_square)
        ):
            return True
    return False

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


def play_game(human_color: bool = chess.WHITE):
    board = chess.Board()
    
    while not board.is_game_over(claim_draw=True):
        if board.turn == human_color:
            move = human_player(board)
        else:
            move = iterative_deepening(board, 0, not human_color) 

        os.system('cls')
        if not board.turn == human_color:
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