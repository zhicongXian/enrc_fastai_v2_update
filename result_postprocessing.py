import pandas as pd

result_folder = "./enrc_results/nr_objects"
import numpy as np

import csv
from itertools import permutations
from pathlib import Path


def best_nmi_mapping(
    csv_path,
    perspectives=( "color","material", "shape"),
    transpose=False,
):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        headers = next(reader)  # Column names: 0;1;2
        matrix = [
            [float(value) for value in row]
            for row in reader
            if row
        ]

    n = len(perspectives)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"Expected a {n} × {n} NMI matrix.")

    # Set True if rows are learned clusterings and columns are perspectives.
    if transpose:
        matrix = list(map(list, zip(*matrix)))
        clustering_ids = [str(i) for i in range(n)]
    else:
        clustering_ids = headers

    # permutation[i] is the clustering assigned to perspective i.
    best = max(
        permutations(range(n)),
        key=lambda permutation: sum(
            matrix[i][permutation[i]] for i in range(n)
        ),
    )

    mapping = {
        perspective: {
            "clustering": clustering_ids[best[i]],
            "nmi": matrix[i][best[i]],
        }
        for i, perspective in enumerate(perspectives)
    }
    mean_nmi = sum(item["nmi"] for item in mapping.values()) / n
    return mapping, mean_nmi


# Process all matching files in this directory.
mean_nmi_list = []
for csv_path in sorted(Path(result_folder).glob("*_nmis.csv")):
    mapping, mean_nmi = best_nmi_mapping(csv_path)

    print(f"\n{csv_path.name}")
    for perspective, result in mapping.items():
        print(
            f"  {perspective:8s} -> clustering {result['clustering']}"
            f" | NMI = {result['nmi']:.4f}"
        )
    print(f"  Best mean NMI = {mean_nmi:.4f}")
    mean_nmi_list.append(mean_nmi)

print(f"Over 10 runs: means {np.mean(mean_nmi_list)}, std: {np.std(mean_nmi_list)} ")