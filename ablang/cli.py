"""Command-line interface to AbLang.

AbLang has separate heavy- and light-chain models.  Input is a CSV file
with a `sequence` column and an optional `id` column.  Per-sequence
results (`seqcoding`, `restore`) are written as CSV; per-residue results
(`rescoding`, `likelihood`) are written as a NumPy `.npz` archive with one
array per sequence, keyed by id.

Weights are downloaded into the installed package on first use, unless
`--model-folder` names a directory that already holds a model.
"""

import argparse
import sys

import numpy as np
import pandas as pd

MODES = ["seqcoding", "rescoding", "likelihood", "restore"]
PER_SEQUENCE_MODES = {"seqcoding", "restore"}
CHAINS = ["heavy", "light"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ablang", description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("input", help="CSV file with a `sequence` column")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output file: CSV for per-sequence modes, .npz for per-residue modes",
    )
    parser.add_argument("--chain", required=True, choices=CHAINS)
    parser.add_argument(
        "--model-folder",
        default="download",
        help="directory holding the chain's model (default: download it "
        "into the installed package)",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="number the sequences with ANARCI and align the output "
        "(`rescoding` and `restore` only; needs ANARCI, which must be "
        "installed from Bioconda)",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="e.g. `cpu` or `cuda`")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    table = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if "sequence" not in table.columns:
        sys.exit("ablang: input lacks a `sequence` column")
    ids = table["id"] if "id" in table.columns else table.index.astype(str)
    sequences = list(table["sequence"])

    # Imported late: loading torch is slow, and `--help` shouldn't wait.
    from .pretrained import pretrained

    model = pretrained(
        args.chain,
        model_folder=args.model_folder,
        ncpu=args.ncpu,
        device=args.device,
    )
    model.freeze()
    result = model(
        sequences, mode=args.mode, align=args.align, splitSize=args.batch_size
    )

    if args.mode in PER_SEQUENCE_MODES:
        values = np.asarray(result)
        if values.ndim == 1:
            frame = pd.DataFrame({args.mode: values})
        else:
            frame = pd.DataFrame(values)
        frame.insert(0, "id", list(ids))
        frame.to_csv(args.output, index=False)
    else:
        np.savez(args.output, **{str(i): np.asarray(r) for i, r in zip(ids, result)})


if __name__ == "__main__":
    main()
