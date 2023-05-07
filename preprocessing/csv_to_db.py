import sqlite3

PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k"]
PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)

class Board:
    def __init__(self, fen: str) -> None:
        self.whitePawns = 0
        self.whiteKnights = 0
        self.whiteBishops = 0
        self.whiteRooks = 0
        self.whiteQueens = 0
        self.whiteKings = 0

        self.blackPawns = 0
        self.blackKnights = 0
        self.blackBishops = 0
        self.blackRooks = 0
        self.blackQueens = 0
        self.blackKings = 0

        self._set_board_fen(fen)
    
    def _set_board_fen(self, fen: str) -> None:
        square_index = 0
        for c in fen:
            if c in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                square_index += int(c)
            elif c.lower() in PIECE_SYMBOLS:
                self._set_piece_at(square_index, PIECE_SYMBOLS.index(c.lower()), c.isupper())
                square_index += 1

    def _set_piece_at(self, square, piece_type, color) -> None:
        mask = 1 << square
        if color == True:
            if piece_type == PAWN:
                self.whitePawns |= mask
            elif piece_type == KNIGHT:
                self.whiteKnights |= mask
            elif piece_type == BISHOP:
                self.whiteBishops |= mask
            elif piece_type == ROOK:
                self.whiteRooks |= mask
            elif piece_type == QUEEN:
                self.whiteQueens |= mask
            else:
                self.whiteKings |= mask
        else:
            if piece_type == PAWN:
                self.blackPawns |= mask
            elif piece_type == KNIGHT:
                self.blackKnights |= mask
            elif piece_type == BISHOP:
                self.blackBishops |= mask
            elif piece_type == ROOK:
                self.blackRooks |= mask
            elif piece_type == QUEEN:
                self.blackQueens |= mask
            else:
                self.blackKings |= mask

    def get_bitmasks(self):
        return f'{self.whitePawns:0>64b}' + \
        f'{self.whiteKnights:0>64b}' + \
        f'{self.whiteBishops:0>64b}' + \
        f'{self.whiteRooks:0>64b}' + \
        f'{self.whiteQueens:0>64b}' + \
        f'{self.whiteKings:0>64b}' + \
        f'{self.blackPawns:0>64b}' + \
        f'{self.blackKnights:0>64b}' + \
        f'{self.blackBishops:0>64b}' + \
        f'{self.blackRooks:0>64b}' + \
        f'{self.blackQueens:0>64b}' + \
        f'{self.blackKings:0>64b}'

con = sqlite3.connect('440m.db')
cur = con.cursor()
cur.execute('DROP TABLE IF EXISTS evaluations')
cur.execute('CREATE TABLE evaluations(id INTEGER PRIMARY KEY, fen TEXT, binary BLOB, eval REAL)')

positions = []
with open('440m.csv', 'r') as f:
    id = 0
    for line in f:
        [fen, ev] = line.split(',')
        split_fen = fen.split(' ')
        board = Board(split_fen[0])
        bitboards = board.get_bitmasks()
        hmvc = f'{int(split_fen[4]):0>8b}'
        fmvn = f'{int(split_fen[5]):0>16b}' 
        if split_fen[3] == '-':
            eps = '1111110'
        else:
            eps = f'{int(split_fen[3][1]) - 1:0>3b}' + f'{ord(split_fen[3][0]) - 97:0>3b}' + '1'
        color = '0' if split_fen[1] == 'w' else '1'
        castling = ''
        castling += '1' if 'K' in split_fen[2] else '0'
        castling += '1' if 'Q' in split_fen[2] else '0'
        castling += '1' if 'k' in split_fen[2] else '0'
        castling += '1' if 'q' in split_fen[2] else '0'
        binary = bitboards + hmvc + fmvn + eps + color + castling
        positions.append((id, fen, binary, ev))
        id += 1
        if id % 1_000_000 == 0:  
            cur.executemany('INSERT INTO evaluations(id, fen, binary, eval) VALUES(?,?,?,?)',positions)
            con.commit()
            positions = []
            print(str(id/1_000_000))

if positions != []:
    cur.executemany('INSERT INTO evaluations(id, fen, binary, eval) VALUES(?,?,?,?)',positions)
    con.commit()
