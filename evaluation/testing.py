import math
import re
import chess
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from peewee import *
from torch import nn
from collections import OrderedDict
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset

db = SqliteDatabase('test.db')
LABEL_COUNT = 33129

class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  binary = BlobField()
  eval = FloatField()

  class Meta:
    database = db

class EvaluationDataset(Dataset):
  def __init__(self, count):
    self.count = count

  def __len__(self):
    return self.count
  
  def __getitem__(self, idx):
    eval = Evaluations.get(Evaluations.id == idx)
    bin = np.fromiter(map(int, eval.binary), dtype=np.single)
    if type(eval.eval) == str:
        eval.eval = eval.eval.removeprefix('#')
        eval.eval = re.sub(r'\d+', '15', eval.eval) 
    else:
        eval.eval = max(eval.eval, -15)
        eval.eval = min(eval.eval, 15)
    ev = np.array([eval.eval], dtype=np.single)
    return {'binary': bin, 'eval': ev, 'fen': eval.fen}    

class EvaluationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.batch_size = 256
        self.learning_rate = learning_rate
        layers = []
        for i in range(5):
            layers.append((f"linear-{i}", nn.Linear(804, 804)))
            layers.append((f"relu-{i}", nn.ReLU()))

        layers.append((f"linear-{5}", nn.Linear(804, 1)))
        self.save_hyperparameters()
        self.layers = layers
        self.seq = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.seq(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['binary'], batch['eval']
        y_pred = self(x)
        loss = F.l1_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
model = EvaluationModel.load_from_checkpoint(checkpoint_path=r'./model.ckpt', map_location="cpu")
dataset = EvaluationDataset(count=LABEL_COUNT)

piece_values = {
    chess.BISHOP: 330,
    chess.KING: 20_000,
    chess.KNIGHT: 320,
    chess.PAWN: 100,
    chess.QUEEN: 900,
    chess.ROOK: 500,
}

piece_values_simple = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0 
    }

def get_piece_value(piece: chess.Piece) -> int:
    factor = -1 if piece.color == chess.BLACK else 1
    return factor * piece_values.get(piece.piece_type)

def get_piece_value_simple(piece: chess.Piece) -> int:
    factor = -1 if piece.color == chess.BLACK else 1
    return factor * piece_values_simple.get(piece.piece_type)

piece_squared_tables = {
    chess.BISHOP: (
        (-20, -10, -10, -10, -10, -10, -10, -20),
        (-10,   0,   0,   0,   0,   0,   0, -10),
        (-10,   0,   5,  10,  10,   5,   0, -10),
        (-10,   5,   5,  10,  10,   5,   5, -10),
        (-10,   0,  10,  10,  10,  10,   0, -10),
        (-10,  10,  10,  10,  10,  10,  10, -10),
        (-10,   5,   0,   0,   0,   0,   5, -10),
        (-20, -10, -10, -10, -10, -10, -10, -20),
    ),
    chess.KING: (
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-30, -40, -40, -50, -50, -40, -40, -30),
        (-20, -30, -30, -40, -40, -30, -30, -20),
        (-10, -20, -20, -20, -20, -20, -20, -10),
        ( 20,  20,   0,   0,   0,   0,  20,  20),
        ( 20,  30,  10,   0,   0,  10,  30,  20),
    ),
    chess.KNIGHT: (
        (-50, -40, -30, -30, -30, -30, -40, -50),
        (-40, -20,   0,   0,   0,   0, -20, -40),
        (-30,   0,  10,  15,  15,  10,   0, -30),
        (-30,   5,  15,  20,  20,  15,   5, -30),
        (-30,   0,  15,  20,  20,  15,   0, -30),
        (-30,   5,  10,  15,  15,  10,   5, -30),
        (-40, -20,   0,   5,   5,   0, -20, -40),
        (-50, -40, -30, -30, -30, -30, -40, -50),
    ),
    chess.PAWN: (
        (  0,   0,   0,   0,   0,   0,   0,   0),
        ( 50,  50,  50,  50,  50,  50,  50,  50),
        ( 10,  10,  20,  30,  30,  20,  10,  10),
        (  5,   5,  10,  25,  25,  10,   5,   5),
        (  0,   0,   0,  20,  20,   0,   0,   0),
        (  5,  -5, -10,   0,   0, -10,  -5,   5),
        (  5,  10,  10, -20, -20,  10,  10,   5),
        (  0,   0,   0,   0,   0,   0,   0,   0),
    ),
    chess.QUEEN: (
        (-20, -10, -10,  -5,  -5, -10, -10, -20),
        (-10,   0,   0,   0,   0,   0,   0, -10),
        (-10,   0,   5,   5,   5,   5,   0, -10),
        ( -5,   0,   5,   5,   5,   5,   0,  -5),
        (  0,   0,   5,   5,   5,   5,   0,  -5),
        (-10,   5,   5,   5,   5,   5,   0, -10),
        (-10,   0,   5,   0,   0,   0,   0, -10),
        (-20, -10, -10,  -5,  -5, -10, -10, -20),
    ),
    chess.ROOK: (
        (  0,   0,   0,   0,   0,   0,   0,   0),
        (  5,  10,  10,  10,  10,  10,  10,   5),
        ( -5,   0,   0,   0,   0,   0,   0,  -5),
        ( -5,   0,   0,   0,   0,   0,   0,  -5),
        ( -5,   0,   0,   0,   0,   0,   0,  -5),
        ( -5,   0,   0,   0,   0,   0,   0,  -5),
        ( -5,   0,   0,   0,   0,   0,   0,  -5),
        (  0,   0,   0,   5,   5,   0,   0,   0),
    ),
}

piece_squared_tables = {key: tuple(reversed(list(value))) 
                        for key, value in piece_squared_tables.items()}

reversed_piece_squared_tables = {key: tuple([
                                            piece[::-1]
                                            for piece in value][::-1]) 
                                 for key, value in piece_squared_tables.items()}

def get_piece_squared_tables_value(piece: chess.Piece, square: int) -> float:
    factor = -1 if piece.color == chess.BLACK else 1
    row = square // 8
    column = square % 8
    
    if piece.color == chess.WHITE:
        piece_squared_table = piece_squared_tables.get(piece.piece_type)
    else:
        piece_squared_table = reversed_piece_squared_tables.get(piece.piece_type)
        
    return factor * piece_squared_table[row][column]

def piece_square_table_balance(board: chess.Board) -> float:
    piece_value = 0
    for square in range(64):
        piece = board.piece_at(square)
        if not piece:
            continue
        piece_value += get_piece_value(piece)
        piece_value += get_piece_squared_tables_value(piece, square)
    
    return min(max(piece_value/100, -15), 15)

def material_balance(board: chess.Board) -> float:
    piece_value = 0
    for square in range(64):
        piece = board.piece_at(square)
        if not piece:
            continue
        piece_value += get_piece_value_simple(piece)
    return min(max(piece_value, -15), 15)

def avg(lst):
    return sum(lst) / len(lst)


def guess_zero_loss(idx):
    batch = dataset[idx]
    y = torch.tensor(batch['eval'])
    y_pred = torch.zeros_like(y)
    loss = F.l1_loss(y_pred, y)
    return loss

def guess_material_loss(idx):
    batch = dataset[idx]
    board = chess.Board(batch['fen'])
    y = torch.tensor(batch['eval'])
    y_pred = torch.tensor([material_balance(board)])
    loss = F.l1_loss(y_pred, y)
    return loss

def guess_piece_square_loss(idx):
    batch = dataset[idx]
    board = chess.Board(batch['fen'])
    y = torch.tensor(batch['eval'])
    y_pred = torch.tensor([piece_square_table_balance(board)])
    loss = F.l1_loss(y_pred, y)
    return loss

def guess_model_loss(idx):
    batch = dataset[idx]
    x, y = torch.tensor(batch['binary']), torch.tensor(batch['eval'])
    y_pred = model(x)
    loss = F.l1_loss(y_pred, y)
    return loss

# L1 loss
# zero_losses = []
# mat_losses = []
# pst_losses = []
# model_losses = []
# true_value = []
# for i in range(LABEL_COUNT):
#     zero_losses.append(guess_zero_loss(i))
#     mat_losses.append(guess_material_loss(i))
#     pst_losses.append(guess_piece_square_loss(i))
#     model_losses.append(guess_model_loss(i))
# print(f'Guess Zero Avg Loss {avg(zero_losses)}')
# print(f'Guess Material Avg Loss {avg(mat_losses)}')
# print(f'Guess Piece Square Avg Loss {avg(pst_losses)}')
# print(f'Guess Model Avg Loss {avg(model_losses)}')

# RMSE
pred_zero = [0] * LABEL_COUNT 
pred_mat = []
pred_pst = []
pred_model = []
true_value = []
for i in range(LABEL_COUNT):
    batch = dataset[i]
    board = chess.Board(batch['fen'])
    pred_mat.append(material_balance(board))
    pred_pst.append(piece_square_table_balance(board))
    pred_model.append(model(torch.tensor(batch['binary'])).item())
    true_value.append(batch['eval'])
    
zero_RMSE = math.sqrt(mean_squared_error(true_value, pred_zero))
mat_RMSE = math.sqrt(mean_squared_error(true_value, pred_mat))
pst_RMSE = math.sqrt(mean_squared_error(true_value, pred_pst))
model_RMSE = math.sqrt(mean_squared_error(true_value, pred_model))

print(f'Zero RMSE {zero_RMSE}')
print(f'Material RMSE {mat_RMSE}')
print(f'Piece Square Table RMSE {pst_RMSE}')
print(f'ANN RMSE {model_RMSE}')
