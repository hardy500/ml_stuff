
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Engine:
  def __init__(self, model, loss_fn, optimizer, device):
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.device = device

    self.trainloader = None
    self.validloader = None

    self.train_loss = 0
    self.val_loss = 0

  def set_loader(self, trainloader, validloader):
    self.trainloader = trainloader
    self.validloader = validloader

  def train_step(self):
    self.model.train()
    for x, y in self.trainloader:
      x, y = x.to(self.device), y.to(self.device)
      self.optimizer.zero_grad()

      out = self.model(x)
      loss = self.loss_fn(out, y)
      loss.backward()
      self.optimizer.step()

      # accumulate loss
      self.train_loss += loss.item()
    self.train_loss /= len(self.trainloader)

  def eval_step(self):
    self.model.eval()
    with torch.inference_mode():
      for x, y in self.validloader:
        x, y = x.to(self.device), y.to(self.device)
        out = self.model(x)
        loss = self.loss_fn(out, y)
        self.val_loss += loss.item()
      self.val_loss /= len(self.validloader)

  def _set_seed(self):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  def train(self, epochs):
    self._set_seed()

    writer = SummaryWriter()
    self.model.to(self.device)

    for epoch in tqdm(range(epochs)):
      self.train_step()
      self.eval_step()

      writer.add_scalars(
        main_tag="loss",
        tag_scalar_dict={
          "train_loss": self.train_loss,
          "val_loss": self.val_loss
        },
        global_step=epoch
      )

    writer.close()