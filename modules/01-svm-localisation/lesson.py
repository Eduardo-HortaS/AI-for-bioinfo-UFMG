# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: 01-svm-localisation
#     language: python
#     name: 01-svm-localisation
# ---

# %% [markdown]
# # Predicting where a protein lives with a support vector machine
#
# A eukaryotic protein is made in the cytoplasm and then either stays there
# or is sent somewhere else: the nucleus, a mitochondrion, out of the cell.
# Where it ends up is a large part of what it does, and for most sequenced
# proteins nobody has measured it.
#
# In 2001 Hua and Sun showed that a support vector machine (SVM) could
# predict the location of a eukaryotic protein among four compartments using
# nothing but its amino-acid composition, and do it better than the earlier
# composition-based methods. Their tool, SubLoc, became one of the standard
# examples of SVMs in bioinformatics.
#
# This lesson rebuilds that idea on current Swiss-Prot data:
#
# 1. turn each sequence into a fixed-length vector of composition features;
# 2. train a linear SVM as a baseline;
# 3. train an SVM with an RBF kernel and tune it;
# 4. look at what it gets right and wrong, and compare with the 2001 result.
#
# Everything runs in well under a minute.

# %%
import gzip
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

random.seed(0)
os.makedirs("out", exist_ok=True)

# %% [markdown]
# ## The data
#
# `data/localisation.fasta.gz` holds reviewed eukaryotic proteins from
# UniProtKB/Swiss-Prot whose location has been shown experimentally and falls
# into exactly one of four classes: nucleus, cytoplasm, mitochondrion or
# extracellular (secreted). Near-identical proteins have been removed by
# keeping one sequence per UniRef50 cluster, so no two sequences share more
# than about 50% identity. Each header reads `>ACCESSION class CLUSTER`.
#
# The classes are unbalanced, so we take the same number from each. With four
# balanced classes, guessing gets 25%.

# %%
by_class = defaultdict(list)
with gzip.open("data/localisation.fasta.gz", "rt") as f:
    for header, seq in zip(f, f):
        accession, label, cluster = header[1:].split()
        by_class[label].append(seq.strip())

for label, seqs in sorted(by_class.items()):
    print(f"{label:<14}{len(seqs):>5} sequences")

per_class = 1250
sequences, labels = [], []
for label, seqs in sorted(by_class.items()):
    for seq in random.sample(seqs, per_class):
        sequences.append(seq)
        labels.append(label)
labels = np.array(labels)
classes = sorted(by_class)

# %% [markdown]
# ## From sequences to vectors
#
# An SVM needs every example to be a vector of the same length, and proteins
# range from tens to thousands of residues. Hua and Sun used the simplest
# fixed-length summary there is: the fraction of each of the 20 amino acids
# (amino-acid composition, AAC). We add the fraction of each of the 400
# ordered pairs of neighbouring residues (dipeptide composition, DPC), which
# keeps a little information about order. That gives 420 numbers per protein.

# %%
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def composition(seq):
    aac = np.zeros(20)
    dpc = np.zeros((20, 20))
    for a in seq:
        aac[INDEX[a]] += 1
    for a, b in zip(seq, seq[1:]):
        dpc[INDEX[a], INDEX[b]] += 1
    return np.concatenate([aac / len(seq), dpc.ravel() / (len(seq) - 1)])


X = np.array([composition(s) for s in sequences])
print(X.shape)

# %% [markdown]
# Before training anything, it is worth checking that composition differs
# between compartments at all. The heat map shows, for each class, how much
# more (red) or less (blue) of each amino acid it has than the average
# protein in the set, on a log2 scale.
#
# The strongest signal is cysteine (C) in secreted proteins. Outside the cell
# disulfide bonds are stable and help hold the fold together; inside, the
# reducing environment keeps cysteines free, and proteins use fewer of them.
# The other differences are smaller but have plausible explanations:
# mitochondrial proteins carry a little extra arginine (R), as their
# N-terminal transit peptides are rich in it; nuclear proteins have more
# serine, glutamine and proline (S, Q, P), which are common in the flexible,
# disordered regions of transcription factors and other nuclear proteins.
#
# Notice how pale most of the map is. Apart from cysteine, composition
# differs by tens of per cent at most, and that already tells you the
# classifier's job will not be easy.

# %%
mean_aac = np.array([X[labels == c, :20].mean(axis=0) for c in classes])
enrichment = np.log2(mean_aac / X[:, :20].mean(axis=0))

fig, ax = plt.subplots(figsize=(9, 2.8))
im = ax.imshow(enrichment, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(20), list(AMINO_ACIDS))
ax.set_yticks(range(len(classes)), classes)
fig.colorbar(im, ax=ax, label="log2 enrichment")
ax.set_title("Amino-acid composition by compartment")
fig.tight_layout()
fig.savefig("out/composition.png", dpi=120)

# %% [markdown]
# ## Training and test sets
#
# We hold back 20% of the proteins and do not look at them until the end.
# Because the data contain no close homologues, a random split is fair here.
# On redundant data it would not be: a test protein with a near-identical
# twin in the training set is not much of a test.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, stratify=labels, random_state=0
)
print(f"{len(y_train)} training, {len(y_test)} test")

# %% [markdown]
# ## A linear SVM
#
# A linear SVM looks for the hyperplane that separates two classes with the
# widest possible margin. Only the training points closest to the boundary,
# the support vectors, decide where it goes. The parameter `C` sets how much
# a point on the wrong side costs: small `C` tolerates mistakes in exchange
# for a wider, simpler margin; large `C` tries harder to classify every
# training point, at the risk of fitting noise.
#
# SVMs measure distances between vectors, so features with large values
# would dominate. `StandardScaler` rescales each feature to mean 0 and
# standard deviation 1. Putting it in a pipeline means it learns its scaling
# from the training data only.

# %%
linear = make_pipeline(StandardScaler(), LinearSVC(C=0.01))
linear.fit(X_train, y_train)
print(f"linear SVM, test accuracy: {linear.score(X_test, y_test):.3f}")

# %% [markdown]
# ## An SVM with an RBF kernel
#
# Classes are rarely separated by a flat surface. The kernel trick lets an
# SVM draw curved boundaries without ever computing new features: it only
# needs a similarity between pairs of proteins. The radial basis function
# (RBF) kernel uses `exp(-gamma * distance²)`. `gamma` sets how far the
# influence of one training protein reaches: large values give a wiggly
# boundary that hugs individual points, small values a smooth one.
#
# We pick `C` and `gamma` by five-fold cross-validation on the training set:
# split it into five parts, train on four, score on the fifth, rotate, and
# average. The test set stays untouched. scikit-learn's `SVC` handles four
# classes by training one SVM for every pair of classes and letting them
# vote.
#
# This is the slow step: 9 settings × 5 folds = 45 fits, under a minute on a
# typical laptop.

# %%
C_values = [0.1, 1, 10]
gamma_values = [1e-4, 1e-3, 1e-2]
search = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    param_grid={"svc__C": C_values, "svc__gamma": gamma_values},
    cv=5,
    n_jobs=-1,
)
search.fit(X_train, y_train)

results = search.cv_results_
for params, score in zip(results["params"], results["mean_test_score"]):
    print(f"C={params['svc__C']:<5} gamma={params['svc__gamma']:<7} cross-validated accuracy {score:.3f}")
print("best:", search.best_params_)

# The grid is searched with C in the outer loop and gamma in the inner one,
# so the nine scores fold into a C × gamma table.
scores = results["mean_test_score"].reshape(len(C_values), len(gamma_values))
best_row, best_col = np.unravel_index(scores.argmax(), scores.shape)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
im = ax.imshow(scores, cmap="viridis")
for i in range(len(C_values)):
    for j in range(len(gamma_values)):
        ax.text(j, i, f"{scores[i, j]:.3f}", ha="center", va="center",
                color="black" if scores[i, j] > scores.mean() else "white")
ax.add_patch(plt.Rectangle((best_col - 0.5, best_row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))
ax.set_xticks(range(len(gamma_values)), gamma_values)
ax.set_yticks(range(len(C_values)), C_values)
ax.set(xlabel="gamma", ylabel="C", title="Cross-validated accuracy")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("out/grid_search.png", dpi=120)

# %% [markdown]
# Read the heat map before moving on; the best setting is outlined in red. In
# each row, accuracy should rise and then fall as `gamma` grows: too small and
# the boundary is almost flat, too large and it wraps around individual
# training proteins. Larger `C` helps when `gamma` is small, because it lets
# the boundary bend more to fit the training data. If the best cell sat on
# the edge of the grid, the grid would be too narrow and should be extended in
# that direction.

# %% [markdown]
# ## How well does it do?

# %%
best = search.best_estimator_
y_pred = best.predict(X_test)
print(f"RBF SVM, test accuracy: {best.score(X_test, y_test):.3f}")
print(f"support vectors: {best.named_steps['svc'].n_support_.sum()} of {len(y_train)} training proteins")
print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False, xticks_rotation=30)
ax.set_title("RBF SVM on held-out proteins")
fig.tight_layout()
fig.savefig("out/confusion_matrix.png", dpi=120)

# %% [markdown]
# The SVM gets roughly two proteins in three right, against one in four by
# chance. Secreted proteins are the easiest class, then mitochondrial ones:
# both are routed by N-terminal signals (signal peptides and transit
# peptides) that leave a mark on composition. Cytoplasm is the hardest, and
# is confused with both the nucleus and mitochondria. Proteins with no
# targeting signal default to the cytoplasm, many shuttle in and out of the
# nucleus, and a protein's overall composition says little about which side
# of a membrane it spends its time on.
#
# The number of support vectors is also telling. When most of the training
# set ends up on or inside the margin, the classes overlap heavily in this
# feature space, and no choice of `C` or `gamma` will separate them cleanly.
#
# ## Why not 79%?
#
# Hua and Sun reported 79.4% on four eukaryotic classes, and the neural
# network of Reinhardt and Hubbard (1998) reached 66% on the same task.
# Several things differ from our setting:
#
# - Their benchmark was collected from Swiss-Prot in 1998, from a database a
#   small fraction of today's size, and filtered by different rules. Ours
#   keeps one protein per UniRef50 cluster, which removes close relatives;
#   relatives share both composition and location, which makes the problem
#   easier.
# - They estimated accuracy with a jackknife (leave-one-out) test over the
#   whole set rather than a separate test set.
# - Annotation has changed over 25 years. Our labels require experimental
#   evidence, and proteins with more than one known location are excluded.
#
# Removing close homologues costs accuracy, and it is now standard to report
# results on homology-reduced data because they better predict how a method
# will behave on a genuinely new protein. Whenever you read a reported
# accuracy, ask how different the test data were from the training data.
#
# ## Things to try
#
# 1. Use only the 20 amino-acid fractions (`X[:, :20]`). How much do the 400
#    dipeptide features add?
# 2. Compute composition on the first 50 residues and on the rest
#    separately, and use both. Targeting signals sit at the N-terminus, so
#    this should help the mitochondrial and extracellular classes most.
# 3. Compare training and test accuracy for `C=100` and `gamma=0.1`. What
#    does the gap tell you?
# 4. Swap `SVC` for `GaussianNB` (`sklearn.naive_bayes`) or
#    `DecisionTreeClassifier` (`sklearn.tree`), the two other classic
#    methods. Which is closest to the SVM?
# 5. Use all the proteins instead of 1,250 per class, with
#    `SVC(class_weight="balanced")`. What happens to the mitochondrial class?
