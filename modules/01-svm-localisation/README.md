# 01 · SVM: where does a protein live?

Predict whether a eukaryotic protein sits in the nucleus, the cytoplasm, a
mitochondrion or outside the cell, using only its amino-acid composition and
a support vector machine. This is a modern reproduction of SubLoc (Hua and
Sun, 2001), one of the classic applications of SVMs in bioinformatics.

## Running

Open `lesson.py` in JupyterLab, where it opens as a notebook, or run it from a
terminal in this folder:

```
uv run lesson.py
```

It takes about a minute. Figures are saved in `out/`.

## Data

`data/localisation.fasta.gz` contains 10,878 reviewed eukaryotic proteins from
UniProtKB/Swiss-Prot release 2026_04:

| Class | Proteins |
|---|---|
| nucleus | 4,083 |
| cytoplasm | 3,248 |
| extracellular | 2,165 |
| mitochondrion | 1,382 |

Each protein has a single, experimentally supported location, and only one
protein per UniRef50 cluster is kept. `tools/build_localisation_set.py` in
the repository root documents the exact rules and rebuilds the file.

UniProt data are available under the
[Creative Commons Attribution 4.0 licence](https://www.uniprot.org/help/license).
See [how to cite UniProt](https://www.uniprot.org/help/publications).

## References

- Hua S, Sun Z (2001). Support vector machine approach for protein
  subcellular localization prediction. *Bioinformatics* 17(8):721–728.
  [doi:10.1093/bioinformatics/17.8.721](https://doi.org/10.1093/bioinformatics/17.8.721)
- Reinhardt A, Hubbard T (1998). Using neural networks for prediction of the
  subcellular location of proteins. *Nucleic Acids Research* 26(9):2230–2236.
  [doi:10.1093/nar/26.9.2230](https://doi.org/10.1093/nar/26.9.2230),
  free on [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC147531/).
- Ben-Hur A, Ong CS, Sonnenburg S, Schölkopf B, Rätsch G (2008). Support
  vector machines and kernels for computational biology. *PLoS Computational
  Biology* 4(10):e1000173.
  [doi:10.1371/journal.pcbi.1000173](https://doi.org/10.1371/journal.pcbi.1000173).
  A readable introduction to SVMs aimed at biologists.
- scikit-learn user guide:
  [Support vector machines](https://scikit-learn.org/stable/modules/svm.html).
