import torch
from torch import nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import pandas as pd

def model_setup():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  weights = models.ResNet50_Weights.DEFAULT
  model = models.resnet50(weights=weights)

  for param in model.parameters():
    param.requires_grad = False

  n_features = model.fc.in_features
  n_cls = 10

  model.fc = nn.Linear(n_features, n_cls)
  model.load_state_dict(torch.load('model/model_params.pth', map_location=torch.device(device))) 
  return model

def predict(x):
  x = Image.fromarray(x)
  model =  model_setup()

  transform = transforms.Compose([
      transforms.Resize(232),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
      )
  ])

  x = transform(x)
  model.eval()
  with torch.inference_mode():
    assert isinstance(x, torch.Tensor), "Input is not a tensor"
    out = model(x.unsqueeze(0))
    probs = torch.softmax(out, 1).flatten()

  cls_labels = pd.read_csv('cls_labels.csv', header=None, index_col=False)[0].values
  return {label: float(prob) for (label, prob) in zip(cls_labels, probs)}
