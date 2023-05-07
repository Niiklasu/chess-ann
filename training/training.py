import re, torch, chess
import time
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from peewee import *
from random import randrange
from torch import nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

db = SqliteDatabase('440m.db')
LABEL_COUNT = 480_715_233

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
    return {'binary': bin, 'eval': ev}    

class EvaluationModel(pl.LightningModule):
  def __init__(self,learning_rate=1e-3):
    super().__init__()
    self.batch_size = 1024
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

  def train_dataloader(self):
    dataset = EvaluationDataset(count=LABEL_COUNT)
    return DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)


if __name__ == '__main__':
        dataset = EvaluationDataset(count=LABEL_COUNT)
        torch.set_float32_matmul_precision("medium")

        design = '' 
        model = EvaluationModel()
        for (name, layer) in model.layers:
          if 'linear' in name:
            layer: nn.Linear
            design += str(layer.in_features) + '-' +str(layer.out_features) + ', '
        design = design.strip(', ')
        version_name = f'batch_size={model.batch_size}-design={design}'
        logger = pl.loggers.TensorBoardLogger("lightning_logs", name="480M", version=version_name)
        trainer = pl.Trainer(precision=32, max_epochs=2, logger=logger)
        trainer.fit(model)