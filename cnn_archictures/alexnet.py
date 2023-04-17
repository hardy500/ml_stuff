#%%
import torch
from torch import nn

# input size should be: (bs x 3 x  227 x 227)
# The original paper states an image size (bs x 3 x 224 x 224)
# But dimension of the first conv layer does not lead to 55x55
x = torch.rand(1, 3, 227, 227)

#%%

# paper: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
# TODO: finish the classifier part and clean this up

conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=96,
    kernel_size=11,
    stride=4
)

conv2 = nn.Conv2d(
    in_channels=96,
    out_channels=256,
    kernel_size=5,
    padding=2
)

conv3 = nn.Conv2d(
    in_channels=256,
    out_channels=384,
    kernel_size=3,
    padding=1
)

conv4 = nn.Conv2d(
    in_channels=384,
    out_channels=384,
    kernel_size=3,
    padding=1
)

conv5 = nn.Conv2d(
    in_channels=384,
    out_channels=256,
    kernel_size=3,
    padding=1
)

fc1 = nn.Linear(in_features=(256*6*6), out_features=4096)
fc2 = nn.Linear(in_features=4096, out_features=4096)
fc3 = nn.Linear(in_features=4096, out_features=10000)

relu = nn.ReLU()
lrn = nn.LocalResponseNorm(size=5, alpha=10**-4, beta=0.75, k=2)
pool = nn.MaxPool2d(kernel_size=3, stride=2)

# conv1
out = conv1(x)   # (96x55x55)
out = relu(out)
out = lrn(out)
out = pool(out)  # (96x27x27)

# conv2
out = conv2(out)
out = relu(out)
out = lrn(out)
out = pool(out)  # (256x13x13)

# conv3
out = conv3(out) # (384x13x13)
out = relu(out)

# conv4
out = conv4(out) # (384x13x13)
out = relu(out)

# conv5
out = conv5(out) # (256x13x13)
out = relu(out)
out = pool(out)  # (256x6x6)