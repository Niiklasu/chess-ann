import math
import re
import bot
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


def guess_model_loss(idx):
    batch = dataset[idx]
    x, y = torch.tensor(batch['binary']), torch.tensor(batch['eval'])
    y_pred = model(x)
    loss = F.l1_loss(y_pred, y)

    # if loss > 10:
    #    print(batch['fen'])
    return loss


for i in range(1000):
    guess_model_loss(i)

print(bot.evaluate(chess.Board('r1bqk1nr/ppp2ppp/2np4/b7/2B1P3/1QP2N2/P4PPP/RNB2RK1 w - - 0 1')))
# print(guess_model_loss(8).item(), dataset[8]['eval'], dataset[8]['fen'])
# print(guess_model_loss(9).item(), dataset[9]['eval'], dataset[9]['fen'])