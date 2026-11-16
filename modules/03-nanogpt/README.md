# 03 · nanoGPT: a tiny language model

Train a 0.8-million-parameter GPT on a million characters of Shakespeare, one
character at a time, then train the same model on the protein sequences from
module 01. Along the way, the lesson covers what a language model predicts,
tokenisation, attention, sampling, and why fluent output is not the same as
correct output. It ends with notes on scale, assistants, over-fitting,
leakage and data provenance, and a list of free material for further study.

## Running

Open `lesson.py` in JupyterLab, where it opens as a notebook, or run it from a
terminal in this folder:

```
uv run lesson.py
```

The two training runs take one to three minutes each on a laptop CPU. The
lesson calls the nanoGPT scripts for you. To run them by hand, work inside
`nanogpt/`:

```
cd nanogpt
python data/shakespeare_char/prepare.py
python train.py config/cpu_shakespeare.py
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

## What is in `nanogpt/`

The upstream files are copied unmodified from
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) at commit
[`3adf61e`](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291),
under the MIT licence in `nanogpt/LICENSE`: `model.py`, `train.py`,
`sample.py`, `configurator.py`, `config/train_shakespeare_char.py` and
`data/shakespeare_char/prepare.py`. The upstream repository is no longer
developed, which suits a teaching copy.

Added for this course:

- `config/cpu_shakespeare.py`: the laptop settings from the nanoGPT README
  as a config file, with `dtype = 'float32'` so that CPU training does not
  fall back to half precision.
- `config/cpu_protein.py`: the same settings for the protein data.
- `data/protein_char/prepare.py`: turns the module 01 FASTA file into
  training data, splitting by UniRef50 cluster.
- `data/shakespeare_char/input.txt`: the tiny Shakespeare text from
  [karpathy/char-rnn](https://github.com/karpathy/char-rnn), included so
  that nothing is downloaded at run time. Shakespeare is in the public domain.

`train.py` prints a `FutureWarning` about `torch.cuda.amp.GradScaler`. It is
harmless and left alone so that the upstream code stays untouched.

## References

Papers cited in the lesson (all free to read):

- Vaswani A et al. (2017). Attention is all you need.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Sennrich R, Haddow B, Birch A (2016). Neural machine translation of rare
  words with subword units. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Su J et al. (2021). RoFormer: enhanced transformer with rotary position
  embedding. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Holtzman A et al. (2019). The curious case of neural text degeneration.
  [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- Kaplan J et al. (2020). Scaling laws for neural language models.
  [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann J et al. (2022). Training compute-optimal large language models.
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Ouyang L et al. (2022). Training language models to follow instructions
  with human feedback. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Ferruz N, Schmidt S, Höcker B (2022). ProtGPT2 is a deep unsupervised
  language model for protein design. *Nature Communications* 13:4348.
  [doi:10.1038/s41467-022-32007-7](https://doi.org/10.1038/s41467-022-32007-7)
- Madani A et al. (2020). ProGen: language modeling for protein generation.
  [arXiv:2004.03497](https://arxiv.org/abs/2004.03497)
- Lin Z et al. (2022). Evolutionary-scale prediction of atomic level protein
  structure with a language model.
  [bioRxiv 10.1101/2022.07.20.500902](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)

Further study:

- Karpathy A. [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html),
  with code in [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture).
- Karpathy A. [minbpe](https://github.com/karpathy/minbpe), a minimal BPE tokeniser.
- Zhang A, Lipton ZC, Li M, Smola AJ. [Dive into Deep Learning](https://d2l.ai).
- Prince SJD. [Understanding Deep Learning](https://udlbook.github.io/udlbook/).
- Jurafsky D, Martin JH. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), 3rd edition draft.
- Harvard NLP. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
- Alammar J. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
- Hugging Face. [LLM Course](https://huggingface.co/learn/llm-course).
- 3Blue1Brown. [Neural networks](https://www.3blue1brown.com/topics/neural-networks).
