#%%
import torch
from torch import nn

# paper: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

# input size should be: (bs x 3 x  227 x 227)
# The original paper states an image size (bs x 3 x 224 x 224)
# But dimension of the first conv layer does not lead to 55x55

class AlexNet(nn.Module):
  def __init__(self, n_classes=10_000):
    super().__init__()

    self.featurizer = nn.Sequential(
      # conv1
      nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, alpha=10**-4, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=2),
      # conv2
      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, alpha=10**-4, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=2),
      # conv3
      nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
      nn.ReLU(),
      # conv4
      nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
      nn.ReLU(),
      # conv5
      nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )

    self.classifier = nn.Sequential(
      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=(256*6*6), out_features=4096, bias=False),
      nn.ReLU(),

      nn.Dropout(p=0.5, inplace=True),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(),

      nn.Linear(in_features=4096, out_features=n_classes, bias=False),
    )

    # Init weight and bias
    self.init_w_b()

  def forward(self, x):
    out = self.featurizer(x)
    out = self.classifier(out.reshape(1, -1))
    return out

  def init_w_b(self):
    for layer in self.featurizer:
      if isinstance(layer, nn.Conv2d):
        # conv weight: zero-mean Gaussian distribution with standard deviation 0.01
        nn.init.normal_(layer.weight, mean=0, std=0.01)
        nn.init.constant_(layer.bias, 0)

      # conv bias: l2, l4, l5 = 1
      nn.init.constant_(self.featurizer[4].bias, 1)
      nn.init.constant_(self.featurizer[10].bias, 1)
      nn.init.constant_(self.featurizer[12].bias, 1)

      # fc: l2 = 1
      nn.init.constant_(self.classifier[4].bias, 1)

if __name__ == "__main__":
  x = torch.rand(1, 3, 227, 227)
  model = AlexNet()
  print(model(x).shape)

