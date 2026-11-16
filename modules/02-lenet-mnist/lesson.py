# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: 02-lenet-mnist
#     language: python
#     name: 02-lenet-mnist
# ---

# %% [markdown]
# # Reading handwritten digits with LeNet-5
#
# In 1980 Kunihiko Fukushima, at the research laboratories of the Japanese
# broadcaster NHK, described the neocognitron: a layered network, inspired by
# the visual cortex, in which early layers detect small local features and
# later layers combine them, with each feature detector repeated across the
# whole image. It could recognise handwritten digits even when they were
# shifted or slightly deformed.
#
# The neocognitron was not trained by gradient descent. In 1989 Yann LeCun
# and colleagues at Bell Labs combined the same architectural ideas with
# backpropagation to read handwritten ZIP codes, and in 1998 published
# LeNet-5, which read cheques in production. These are convolutional neural
# networks, and the same principles are at work in most image models today,
# including those used on microscopy images and protein structure maps.
#
# LeNet-5 is small enough to train on a laptop CPU in about a minute. In
# this lesson we will:
#
# 1. load MNIST, the classic data set of 70,000 handwritten digits;
# 2. build LeNet-5 in PyTorch;
# 3. train it and check it on digits it has never seen;
# 4. look at its mistakes and at what its first layer has learnt.

# %%
import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
os.makedirs("out", exist_ok=True)

# %% [markdown]
# ## The data
#
# MNIST was assembled by LeCun, Cortes and Burges from forms filled in by US
# Census Bureau employees and high-school students: 60,000 training and
# 10,000 test images, each 28×28 pixels in greyscale with the digit centred.
#
# The files use a simple binary format (IDX): a short header followed by raw
# bytes, one per pixel. Images have a 16-byte header, labels an 8-byte one,
# so NumPy can read them directly.

# %%
def read_images(path):
    with gzip.open(path) as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)


def read_labels(path):
    with gzip.open(path) as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


# Scale pixels to [0, 1]. The extra dimension of size 1 is the colour
# channel: one for greyscale, three for RGB.
x_all = torch.tensor(read_images("data/train-images-idx3-ubyte.gz"), dtype=torch.float32) / 255
y_all = torch.tensor(read_labels("data/train-labels-idx1-ubyte.gz"), dtype=torch.long)
x_test = torch.tensor(read_images("data/t10k-images-idx3-ubyte.gz"), dtype=torch.float32) / 255
y_test = torch.tensor(read_labels("data/t10k-labels-idx1-ubyte.gz"), dtype=torch.long)

# Keep 5,000 training images aside to monitor training. The test set is
# used once, at the end.
order = torch.randperm(len(x_all))
x_val, y_val = x_all[order[:5000]], y_all[order[:5000]]
x_train, y_train = x_all[order[5000:]], y_all[order[5000:]]
print(f"train {tuple(x_train.shape)}, validation {tuple(x_val.shape)}, test {tuple(x_test.shape)}")

fig, axes = plt.subplots(2, 8, figsize=(8, 2.4))
for ax, image, label in zip(axes.flat, x_train, y_train):
    ax.imshow(image[0], cmap="gray_r")
    ax.set_title(int(label))
    ax.axis("off")
fig.tight_layout()
fig.savefig("out/digits.png", dpi=120)

# %% [markdown]
# ## The network
#
# A convolutional layer slides a small filter (here 5×5 pixels) over the
# image and records how strongly each patch matches it, producing a feature
# map. The same filter is used at every position, so a stroke detector
# learnt in one corner works everywhere, and the layer needs very few
# weights. Pooling then halves the resolution, which makes the next layer
# less sensitive to exactly where a feature appeared.
#
# LeNet-5 stacks these:
#
# | Layer | Output | What it does |
# |---|---|---|
# | C1 convolution | 6 × 28 × 28 | 6 filters of 5×5 over the (padded) image |
# | S2 pooling | 6 × 14 × 14 | average of each 2×2 block |
# | C3 convolution | 16 × 10 × 10 | 16 filters of 5×5 over all 6 maps |
# | S4 pooling | 16 × 5 × 5 | average of each 2×2 block |
# | F5, F6 | 120, then 84 | fully connected layers |
# | Output | 10 | one score per digit |
#
# We follow the original closely but simplify a few details: the 1998
# network connected each C3 map to only some of the S2 maps, gave its
# pooling layers trainable weights, and used a different kind of output
# unit. None of this changes the idea.

# %%
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.f5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(torch.tanh(self.c1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.c3(x)), 2)
        x = x.flatten(1)
        x = torch.tanh(self.f5(x))
        x = torch.tanh(self.f6(x))
        return self.output(x)


model = LeNet5()
print(model)
print(f"{sum(p.numel() for p in model.parameters()):,} trainable parameters")

# %% [markdown]
# About 60,000 parameters, as in the paper. For comparison, a network with
# one fully connected layer of 100 units on the raw pixels already needs
# 78,500, and knows nothing about which pixels are neighbours.
#
# ## Training
#
# Training repeats a simple loop over small batches of images:
#
# 1. run the batch through the network to get ten scores per image;
# 2. measure how wrong they are with the cross-entropy loss, which is low
#    when the correct digit gets a high probability;
# 3. backpropagation computes how the loss changes with every weight;
# 4. the optimiser nudges each weight in the direction that lowers the loss.
#
# One pass over the training set is an epoch. We use the Adam optimiser,
# a standard modern choice; the 1998 paper used a second-order variant of
# stochastic gradient descent.

# %%
def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        correct = (model(x).argmax(dim=1) == y).float().mean().item()
    model.train()
    return correct


optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 64
history = []

for epoch in range(1, 11):
    total_loss = 0.0
    for batch in torch.randperm(len(x_train)).split(batch_size):
        loss = F.cross_entropy(model(x_train[batch]), y_train[batch])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(batch)
    history.append((total_loss / len(x_train), accuracy(model, x_val, y_val)))
    print(f"epoch {epoch:2d}  training loss {history[-1][0]:.4f}  validation accuracy {history[-1][1]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
epochs = range(1, len(history) + 1)
ax1.plot(epochs, [h[0] for h in history], marker="o")
ax1.set(xlabel="epoch", ylabel="training loss")
ax2.plot(epochs, [h[1] for h in history], marker="o")
ax2.set(xlabel="epoch", ylabel="validation accuracy")
fig.tight_layout()
fig.savefig("out/training.png", dpi=120)

# %% [markdown]
# Look at the two curves together. The training loss falls through every
# epoch, but validation accuracy stops improving after about five. From
# then on the network is getting better at the images it has seen without
# getting better at new ones. That is the start of over-fitting, and it is
# why we keep data aside to watch.
#
# ## The test set
#
# Now, and only now, we measure performance on the 10,000 test digits.
# Expect a little over 1% errors. LeNet-5 made 0.95% in 1998, with longer
# training and the refinements mentioned above.

# %%
test_accuracy = accuracy(model, x_test, y_test)
print(f"test accuracy {test_accuracy:.4f}, {round((1 - test_accuracy) * len(y_test))} mistakes out of {len(y_test)}")

# %% [markdown]
# ## Mistakes
#
# The errors are often instructive: many are digits a person would hesitate
# over too, and a few are arguably labelled wrongly.

# %%
model.eval()
with torch.no_grad():
    predicted = model(x_test).argmax(dim=1)
wrong = (predicted != y_test).nonzero().flatten()

fig, axes = plt.subplots(2, 8, figsize=(8, 2.6))
for ax, i in zip(axes.flat, wrong):
    ax.imshow(x_test[i, 0], cmap="gray_r")
    ax.set_title(f"{int(y_test[i])} → {int(predicted[i])}", fontsize=9)
    ax.axis("off")
fig.suptitle("true → predicted")
fig.tight_layout()
fig.savefig("out/mistakes.png", dpi=120)

# %% [markdown]
# ## What the first layer learnt
#
# Nobody told C1 what to look for. Its six 5×5 filters start as random
# numbers and are shaped only by the training loss. Plotted below (top row),
# most look like detectors for edges or strokes at particular angles. The
# bottom row shows the six feature maps they produce for one test digit:
# each highlights the parts of the digit that match its filter.

# %%
filters = model.c1.weight.detach()[:, 0]
with torch.no_grad():
    maps = torch.tanh(model.c1(x_test[:1]))[0]

fig, axes = plt.subplots(2, 6, figsize=(8, 3))
for k in range(6):
    axes[0, k].imshow(filters[k], cmap="RdBu_r")
    axes[1, k].imshow(maps[k], cmap="gray_r")
    axes[0, k].axis("off")
    axes[1, k].axis("off")
fig.suptitle(f"C1 filters (top) and their feature maps for a test '{int(y_test[0])}' (bottom)")
fig.tight_layout()
fig.savefig("out/filters.png", dpi=120)

# %% [markdown]
# ## Things to try
#
# 1. Replace `torch.tanh` with `F.relu` and `F.avg_pool2d` with
#    `F.max_pool2d`, the usual choices in modern networks. Does it train
#    faster? End up more accurate?
# 2. Train on only 600 images (1% of the training set). How does the gap
#    between training and validation behave?
# 3. Shift the test images by three pixels (`torch.roll(x_test, 3, dims=3)`)
#    and measure accuracy again. Then build a network without convolutions,
#    `nn.Sequential(nn.Flatten(), nn.Linear(784, 76), nn.Tanh(), nn.Linear(76, 10))`
#    (also about 60,000 parameters), and repeat. Which copes better, and why?
# 4. Draw a digit yourself: 28×28 pixels, white on black, saved as PNG. Load
#    it with `plt.imread`, keep one channel, and ask the model what it is.
#    Expect it to do worse than on MNIST. Why might that be?
