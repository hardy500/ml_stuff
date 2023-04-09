import torch
from torch import nn, optim
import torchvision.models as models
from hyperopt import fmin, tpe, hp
from sklearn.metrics import accuracy_score
from preprocess import get_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the search space for hyperparameters
space = {
    'lr': hp.loguniform('lr', -6, -2),
    #'weight_decay': hp.loguniform('weight_decay', -6, -2),
    #'momentum': hp.uniform('momentum', 0.8, 0.99),
}

def objective(params):
  weights = models.ResNet50_Weights.DEFAULT
  model = models.resnet50(weights=weights)

  for param in model.parameters():
    param.requires_grad = False

  n_features = model.fc.in_features
  n_cls = 10

  model.fc = nn.Linear(n_features, n_cls)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=params['lr'])#, weight_decay=params['weight_decay'])

  trainloader, validloader, _ = get_loader(batch_size=128)

  model.to(device)

  model.train()
  for x, y in trainloader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
      y_true = []
      y_pred = []

      for x, y in validloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred = torch.max(out, 1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    return -accuracy

# run the optimzation algorithm
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=300)
print(best)
