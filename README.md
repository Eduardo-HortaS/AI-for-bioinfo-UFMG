# AI for Bioinformatics (UFMG)

Short, hands-on lessons in machine learning for bioinformatics, packaged as a
container that runs on an ordinary laptop. No GPU is needed.

| Module | Method | Data | Run time |
|---|---|---|---|
| [01-svm-localisation](modules/01-svm-localisation) | Support vector machine (scikit-learn) | Swiss-Prot proteins | about 1 min |
| [02-lenet-mnist](modules/02-lenet-mnist) | Convolutional network, LeNet-5 (PyTorch) | MNIST handwritten digits | about 1 min |
| [03-nanogpt](modules/03-nanogpt) | Transformer language model (nanoGPT) | Shakespeare, Swiss-Prot proteins | 3 to 5 min |

Each module is a single lesson that reads from top to bottom, runs as a
notebook or as a script, and ends with things to try.

## What you need

- A computer with 8 GB of RAM, 4 CPU cores and about 6 GB of free disk space.
- [Docker](https://docs.docker.com/get-started/get-docker/) (Docker Desktop on
  Windows and macOS), or [Apptainer](https://apptainer.org/docs/user/latest/)
  on Linux and HPC clusters.

## Starting the lab

### With Docker

Create an empty folder for your work, open a terminal in it and run:

```
docker run --rm -it -p 127.0.0.1:8888:8888 -v "${PWD}:/home/lab/work" ghcr.io/eduardo-hortas/ai-for-bioinfo-ufmg:latest
```

In the Windows Command Prompt, write `%cd%` instead of `${PWD}`. The first
download is about 600 MB.

Open the link that starts with `http://127.0.0.1:8888/lab?token=` in your
browser. Stop the lab with Ctrl+C in the terminal.

### With Apptainer

```
apptainer build ai-lab.sif docker://ghcr.io/eduardo-hortas/ai-for-bioinfo-ufmg:latest
mkdir ai-lab && cd ai-lab
apptainer run ../ai-lab.sif
```

On a remote machine, Jupyter only listens locally. Reach it from your own
computer with an SSH tunnel, for example
`ssh -L 8888:localhost:8888 user@cluster`, then open the link it prints.

## Working with the lessons

The first time the lab starts, it copies the `modules` folder into your work
folder. Your edits and everything the lessons produce stay there after the
container stops. To start over with clean copies, rename or delete that
`modules` folder and start the lab again.

In JupyterLab, open `modules/<module>/lesson.py`. It opens as a notebook,
already connected to the right environment. Saving writes back to the `.py`
file. You can also run a lesson from a terminal, including JupyterLab's own:

```
cd modules/01-svm-localisation
uv run lesson.py
```

Run one lesson at a time. Each one already uses every CPU core, and two
training at once on a laptop take far more than twice as long.

## Building the image

```
docker build -t ai-for-bioinfo-ufmg .
```

Every module has its own environment, defined by its `pyproject.toml` and
`uv.lock`, so modules can use different versions of Python and of their
libraries. `build_envs.sh` creates all of them with
[uv](https://docs.astral.sh/uv/) and registers each one as a Jupyter kernel.
It also works without containers (so far tested on Linux):

```
sh build_envs.sh
jupyter/.venv/bin/jupyter lab
```

## Adding a module

1. Create a folder `modules/NN-short-name`.
2. Add a `pyproject.toml` with `requires-python` and the dependencies,
   including `ipykernel`, then run `uv lock` in that folder.
3. Write the lesson as `lesson.py` in Jupytext's percent format. Copy the
   header from an existing lesson and set the kernel name to the folder name.
   Lessons read their data relative to their own folder and write everything
   they produce to `out/`.
4. Add a `README.md` covering what the module teaches, how to run it, where
   its data come from and under what licence, and references.

Continuous integration runs every `lesson*.py` inside the image, so a lesson
that breaks or runs over ten minutes fails the build.

Releases are tagged by date, such as `2026.12`. Pushing a tag builds the image
for amd64 and arm64 and publishes it to the GitHub Container Registry.

## Licence

The course material is released under the MIT licence (`LICENSE`). nanoGPT,
in `modules/03-nanogpt/nanogpt`, is by Andrej Karpathy, also under the MIT
licence. Each module's README states the source and licence of its data.
