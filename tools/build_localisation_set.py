# /// script
# requires-python = ">=3.10"
# ///
"""Rebuild modules/01-svm-localisation/data/localisation.fasta.gz from UniProt.

Maintainer-only. Run from the repository root with:

    uv run tools/build_localisation_set.py

Selection, following the four eukaryotic classes of SubLoc (Hua & Sun 2001):

- reviewed (Swiss-Prot), eukaryotic, not fragments;
- every annotated location falls into exactly one of nucleus, cytoplasm,
  mitochondrion or secreted (extracellular), and at least one of them is
  backed by experimental evidence (ECO:0000269);
- no isoform-specific annotations;
- 50 to 1500 residues, standard amino acids only;
- one sequence per UniRef50 cluster (the alphabetically first accession).

Headers are written as ">ACCESSION class UniRef50_ID". The UniProt release
and class counts are printed at the end; copy them into the module README.
"""

import csv
import gzip
import io
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

OUT = Path("modules/01-svm-localisation/data/localisation.fasta.gz")
API = "https://rest.uniprot.org"

# UniProt location prefix -> class label. Sub-locations such as
# "Mitochondrion inner membrane" or "Cytoplasm, cytoskeleton" keep the class
# of their prefix.
CLASSES = {
    "Nucleus": "nucleus",
    "Cytoplasm": "cytoplasm",
    "Mitochondrion": "mitochondrion",
    "Secreted": "extracellular",
}
TERMS = ["SL-0191", "SL-0086", "SL-0173", "SL-0243"]  # same four, as keyword IDs
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
LOCATION = re.compile(r"([A-Z][^.;{}]*?)\s*\{([^}]*)\}")


def get(url, data=None):
    request = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode() if data else None)
    return urllib.request.urlopen(request, timeout=300)


def fetch_candidates():
    query = (
        "reviewed:true AND taxonomy_id:2759 AND fragment:false AND ("
        + " OR ".join(f"cc_scl_term_exp:{t}" for t in TERMS)
        + ")"
    )
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,cc_subcellular_location,sequence",
        "compressed": "true",
    }
    with get(f"{API}/uniprotkb/stream?{urllib.parse.urlencode(params)}") as response:
        release = response.headers["X-UniProt-Release"]
        text = gzip.decompress(response.read()).decode()
    return release, list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def label(annotation):
    """Return the single class of an annotation, or None if it should be dropped."""
    if "[Isoform" in annotation:
        return None
    annotation = re.sub(r"Note=.*", "", annotation).replace("SUBCELLULAR LOCATION:", "")
    labels, experimental = set(), False
    for location, evidence in LOCATION.findall(annotation):
        prefix = location.split(",")[0].strip()
        labels.add(next((c for p, c in CLASSES.items() if prefix.startswith(p)), "other"))
        experimental |= "ECO:0000269" in evidence
    if len(labels) != 1 or "other" in labels or not experimental:
        return None
    return labels.pop()


def uniref50(accessions):
    job = json.load(get(f"{API}/idmapping/run", {"from": "UniProtKB_AC-ID", "to": "UniRef50", "ids": ",".join(accessions)}))["jobId"]
    while json.load(get(f"{API}/idmapping/status/{job}")).get("jobStatus") in ("NEW", "RUNNING"):
        time.sleep(5)
    lines = get(f"{API}/idmapping/stream/{job}?format=tsv").read().decode().splitlines()
    return dict(line.split("\t")[:2] for line in lines[1:])


def main():
    release, rows = fetch_candidates()
    kept = {}
    for row in rows:
        cls, seq = label(row["Subcellular location [CC]"]), row["Sequence"]
        if cls and 50 <= len(seq) <= 1500 and set(seq) <= AMINO_ACIDS:
            kept[row["Entry"]] = (cls, seq)

    clusters = uniref50(kept)
    missing = set(kept) - set(clusters)
    if missing:
        raise SystemExit(f"{len(missing)} accessions did not map to UniRef50, e.g. {sorted(missing)[:5]}")

    seen = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    # mtime=0 keeps the gzip output byte-identical between runs on the same release.
    with open(OUT, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for accession in sorted(kept):
            cluster = clusters[accession]
            if cluster in seen:
                continue
            seen.add(cluster)
            cls, seq = kept[accession]
            counts[cls] += 1
            gz.write(f">{accession} {cls} {cluster}\n{seq}\n".encode())

    print(f"UniProt release {release}")
    print(f"{len(rows)} candidates, {len(kept)} single-location, {sum(counts.values())} after UniRef50")
    for cls, n in counts.most_common():
        print(f"  {cls:<14}{n}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
