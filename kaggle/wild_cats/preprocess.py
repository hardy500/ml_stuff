from torchvision import transforms
from torch.utils.data import DataLoader

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
  def __init__(self, dataset, transform=None):
    self.dataset = dataset
    self.transform = transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    x = Image.open('data/'+self.dataset['filepaths'].values[index])
    y = self.dataset['class id'].values[index]
    if self.transform:
      x = self.transform(x)
    return x, y

def get_data():
  path = 'data/WILDCATS.CSV'
  data = 'data set'
  df = pd.read_csv(path)
  train  = df[df[data]=='train']
  valid  = df[df[data]=='valid']
  test  = df[df[data]=='test']
  return train, valid, test

def create_loader(data, batch_size, transform=None, shuffle=True):
  d = CustomDataset(data, transform=transform)
  loader = DataLoader(d, batch_size=batch_size, shuffle=shuffle)
  return loader

def get_loader(batch_size=64):
  transform = transforms.Compose([
      transforms.Resize(232),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
      )
  ])

  train, valid, test = get_data()

  trainloader = create_loader(train, batch_size=batch_size, transform=transform)
  validloader = create_loader(valid, batch_size=batch_size, transform=transform)
  testloader = create_loader(test, batch_size=batch_size, transform=transform, shuffle=False)

  return trainloader, validloader, testloader
