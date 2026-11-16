# 02 · LeNet-5: reading handwritten digits

Build and train LeNet-5, the convolutional network LeCun and colleagues
published in 1998, on the MNIST handwritten digits. The lesson covers how
convolution and pooling work, how a network is trained, how to tell when it
starts to over-fit, and what its first layer learns to look for.

Its lineage goes back to the neocognitron, which Kunihiko Fukushima developed
at NHK's research laboratories in Tokyo in 1980 to recognise handwritten
characters regardless of small shifts in position.

## Running

Open `lesson.py` in JupyterLab, where it opens as a notebook, or run it from a
terminal in this folder:

```
uv run lesson.py
```

Training takes about a minute and reaches roughly 98.7% test accuracy. Figures
are saved in `out/`.

## Data

`data/` holds the four original MNIST files (60,000 training and 10,000 test
images, 28×28 pixels), created by Yann LeCun, Corinna Cortes and Christopher
J.C. Burges from NIST handwriting data. They were downloaded from PyTorch's
mirror (`https://ossci-datasets.s3.amazonaws.com/mnist/`) and match the MD5
checksums torchvision uses for them:

```
f68b3c2dcbeaaa9fbdd348bbdeb94873  train-images-idx3-ubyte.gz
d53e105ee54ea40749a09fcbcd1e9432  train-labels-idx1-ubyte.gz
9fb629c4189551a2d022fa330f9573f3  t10k-images-idx3-ubyte.gz
ec29112dd5afa0611ce80d1b7f02629c  t10k-labels-idx1-ubyte.gz
```

The archived MNIST home page states no licence. The data set is commonly
listed as CC BY-SA 3.0, and has been freely redistributed for research and
teaching for decades.

## References

- Fukushima K (1980). Neocognitron: a self-organizing neural network model
  for a mechanism of pattern recognition unaffected by shift in position.
  *Biological Cybernetics* 36(4):193–202.
  [doi:10.1007/BF00344251](https://doi.org/10.1007/BF00344251)
- LeCun Y, Boser B, Denker JS, Henderson D, Howard RE, Hubbard W, Jackel LD
  (1989). Backpropagation applied to handwritten zip code recognition.
  *Neural Computation* 1(4):541–551.
  [doi:10.1162/neco.1989.1.4.541](https://doi.org/10.1162/neco.1989.1.4.541)
- LeCun Y, Bottou L, Bengio Y, Haffner P (1998). Gradient-based learning
  applied to document recognition. *Proceedings of the IEEE*
  86(11):2278–2324. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791),
  [PDF on the author's site](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).
- Zhang A, Lipton ZC, Li M, Smola AJ. *Dive into Deep Learning*,
  [section on LeNet](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html).
- PyTorch documentation: [torch.nn](https://docs.pytorch.org/docs/stable/nn.html).
