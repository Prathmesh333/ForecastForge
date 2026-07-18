"""Train and serialize the reusable lifecycle/peer-prior artifact."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from .boosting import fit_boosting_bundle
from .ingest import load_datasets
from .model import build_model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./datasets")
    parser.add_argument("--model-path", default="./pickle/model.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, quality = load_datasets(args.data_dir)
    artifact = build_model_artifact(data, quality)
    boosting_bundle, boosting_diagnostics = fit_boosting_bundle(data)
    artifact["boosting_training"] = boosting_diagnostics
    if boosting_bundle is not None:
        artifact["boosting_bundle"] = boosting_bundle
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Saved {artifact['model_kind']} artifact with {artifact['training_rows']:,} rows "
        f"and {artifact['training_campaigns']} campaigns to {model_path}"
    )
    print(
        "Boosting challenger: "
        f"{boosting_diagnostics.get('status')} "
        f"({boosting_diagnostics.get('training_rows', 0):,} supervised rows)"
    )


if __name__ == "__main__":
    main()
