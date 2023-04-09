#%%
import torch
from torch import nn, optim
import torchvision.models as models
from preprocess import get_loader
from engine import Engine

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

for param in model.parameters():
  param.requires_grad = False

n_features = model.fc.in_features
n_cls = 10

model.fc = nn.Linear(n_features, n_cls)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
trainloader, validloader, testloader = get_loader(batch_size=128)

#%%
engine = Engine(model, loss_fn, optimizer, device)
engine.set_loader(trainloader, validloader)
engine.train(100)

torch.save(engine.model.state_dict(), 'model/model_params.pth')
