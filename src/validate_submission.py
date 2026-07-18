"""Fail-fast checks for the offline hackathon submission bundle."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .boosting import BUNDLE_KIND, BUNDLE_VERSION, BOOSTER_FORMAT
from .constants import DEFAULT_HORIZONS, MODEL_KIND, OUTPUT_COLUMNS, SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], root: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _find_usable_bash() -> str | None:
    candidates: list[Path] = []
    if os.name == "nt":
        for base in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        ):
            if base:
                candidates.extend(
                    [
                        Path(base) / "Git" / "bin" / "bash.exe",
                        Path(base) / "Programs" / "Git" / "bin" / "bash.exe",
                    ]
                )
    discovered = shutil.which("bash")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _check_required_files(root: Path, errors: list[str], passes: list[str]) -> None:
    required = (
        "run.sh",
        "run.ps1",
        "requirements.txt",
        "README.md",
        "TECHNICAL_DOCUMENTATION.md",
        "ARCHITECTURE.md",
        "DEMO_RUNBOOK.md",
        "MODEL_CARD.md",
        "pickle/model.pkl",
        "src/predict.py",
        "src/forecast.py",
        "src/ai_analyst.py",
        "data/google_sample_campaign_stats.csv",
        "data/meta_sample_campaign_stats.csv",
        "data/microsoft_sample_campaign_stats.csv",
    )
    missing = [path for path in required if not (root / path).is_file()]
    if missing:
        errors.append(f"Missing required files: {missing}")
    else:
        passes.append("required files are present")

    sample_files = [
        path
        for path in (root / "data").glob("*.csv")
        if path.name.lower() not in {"future_budgets.csv", "example_budget_plan.csv"}
    ]
    if not sample_files or any(path.stat().st_size == 0 for path in sample_files):
        errors.append("Public data directory needs non-empty usable CSV inputs")
    else:
        passes.append(f"public sample has {len(sample_files)} usable CSV files")


def _check_requirements(root: Path, errors: list[str], passes: list[str]) -> None:
    lines = [
        line.strip()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pin = re.compile(r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+$")
    bad = [line for line in lines if not pin.fullmatch(line)]
    if bad:
        errors.append(f"Every dependency must be exactly pinned; invalid lines: {bad}")
    else:
        passes.append(f"{len(lines)} dependencies are exactly pinned")
    if "xgboost==3.2.0" not in {line.casefold() for line in lines}:
        errors.append("The committed booster requires the exact xgboost==3.2.0 pin")


def _check_model(root: Path, errors: list[str], passes: list[str]) -> None:
    path = root / "pickle" / "model.pkl"
    if not path.exists():
        return
    if path.stat().st_size >= 95 * 1024 * 1024:
        errors.append("model.pkl is too close to the GitHub 100 MB file limit")
    prefix = path.read_bytes()[:80]
    if prefix.startswith(b"version https://git-lfs.github.com/spec"):
        errors.append("model.pkl is a Git LFS pointer, not the model artifact")
        return
    try:
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
    except Exception as error:
        errors.append(f"model.pkl cannot be loaded: {error}")
        return
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append("model.pkl schema_version does not match the code")
    if artifact.get("model_kind") != MODEL_KIND:
        errors.append("model.pkl model_kind does not match the code")

    bundle = artifact.get("boosting_bundle")
    if not isinstance(bundle, dict):
        errors.append("model.pkl is missing the pretrained boosting bundle")
    else:
        if bundle.get("bundle_kind") != BUNDLE_KIND:
            errors.append("boosting bundle kind does not match the scorer")
        if bundle.get("bundle_version") != BUNDLE_VERSION:
            errors.append("boosting bundle version does not match the scorer")
        if bundle.get("booster_format") != BOOSTER_FORMAT:
            errors.append("boosting bundle is not stored in the expected UBJ format")
        booster_bytes = bundle.get("booster_bytes")
        if not isinstance(booster_bytes, (bytes, bytearray)) or not booster_bytes:
            errors.append("boosting bundle does not contain a serialized pretrained booster")
        encoder = bundle.get("encoder")
        feature_names = encoder.get("feature_names", []) if isinstance(encoder, dict) else []
        if not feature_names:
            errors.append("boosting bundle is missing its fitted feature contract")
        identity_features = [
            name
            for name in feature_names
            if str(name).casefold().split("=", 1)[0]
            in {"campaign_id", "campaign_name"}
        ]
        if identity_features:
            errors.append(
                "boosting bundle exposes raw campaign identity features: "
                f"{identity_features}"
            )
        training = bundle.get("training", {})
        if set(training.get("horizons", ())) != set(DEFAULT_HORIZONS):
            errors.append("boosting bundle was not trained for exactly 30/60/90-day horizons")

    analog_keys = artifact.get("groups", {}).get("analog", {}).keys()
    unhashed = [key for key in analog_keys if not str(key).startswith("analog_")]
    if unhashed:
        errors.append("model.pkl exposes unhashed campaign analog identifiers")
    pools = [
        len(stats.get("positive_roas", []))
        for groups in artifact.get("groups", {}).values()
        for stats in groups.values()
    ]
    shocks = [
        len(values) for values in artifact.get("channel_shocks", {}).values()
    ]
    if (pools and max(pools) > 21) or (shocks and max(shocks) > 21):
        errors.append("model.pkl contains over-detailed empirical samples; retrain compact artifact")
    if not errors:
        passes.append(
            "model artifact and pretrained boosting bundle load, pass privacy checks, "
            f"and are {path.stat().st_size / 1024:.0f} KiB"
        )


def _check_run_scripts(root: Path, errors: list[str], warnings: list[str], passes: list[str]) -> None:
    path = root / "run.sh"
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("run.sh must not contain a UTF-8 BOM")
    if not raw.startswith(b"#!/usr/bin/env bash\n"):
        errors.append("run.sh must start with an LF-terminated bash shebang")
    if b"\r\n" in raw:
        errors.append("run.sh must use LF line endings")
    bash = _find_usable_bash()
    if bash:
        result = _run([bash, "-n", str(path)], root, timeout=30)
        combined = (result.stdout + result.stderr).replace("\x00", "").strip()
        if result.returncode:
            errors.append(f"run.sh fails bash syntax check: {combined}")
        else:
            passes.append(f"run.sh passes bash syntax validation with {Path(bash).name}")
    else:
        warnings.append("bash is unavailable; skipped bash -n")


def _tracked_files(root: Path) -> list[str]:
    result = _run(["git", "ls-files", "-z"], root, timeout=30)
    if result.returncode:
        return []
    return [value for value in result.stdout.split("\0") if value]


def _check_git_index(root: Path, errors: list[str], warnings: list[str], passes: list[str]) -> None:
    if not (root / ".git").exists():
        errors.append("Git repository is not initialized")
        return
    tracked = _tracked_files(root)
    if not tracked:
        errors.append("Git index is empty; stage the intended submission files")
        return

    forbidden: list[str] = []
    for path in tracked:
        lower = path.replace("\\", "/").lower()
        if (
            lower.startswith("datasets/")
            or lower.endswith(".pdf")
            or lower in {
                ".env",
                ".streamlit/secrets.toml",
                "final_winning_idea.md",
                "hackathon_ideas.md",
            }
            or (lower.startswith("output/") and lower != "output/.gitkeep")
            or any(
                fnmatch.fnmatch(lower, pattern)
                for pattern in ("*.log", "*.tmp", "*/__pycache__/*", "*.pyc")
            )
        ):
            forbidden.append(path)
    if forbidden:
        errors.append(f"Confidential or generated files are tracked: {forbidden}")

    mode = _run(["git", "ls-files", "--stage", "--", "run.sh"], root, timeout=30)
    if not mode.stdout.startswith("100755 "):
        errors.append("run.sh must be staged with executable mode 100755")
    else:
        passes.append("run.sh is tracked as executable")

    absolute_needles = [
        "D:" + "\\Hackathon",
        "C:" + "\\Users",
        "/" + "Users/",
    ]
    secret_patterns = (
        re.compile(r"\bsk-" + r"[A-Za-z0-9_-]{16,}\b"),
        re.compile(r"\bAI" + r"za[A-Za-z0-9_-]{30,}\b"),
    )
    leaks: list[str] = []
    text_suffixes = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".sh", ".ps1", ".csv"}
    for relative in tracked:
        path = root / relative
        if not path.is_file() or path.suffix.lower() not in text_suffixes or path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(needle.casefold() in text.casefold() for needle in absolute_needles):
            leaks.append(f"{relative}: absolute local path")
        if any(pattern.search(text) for pattern in secret_patterns):
            leaks.append(f"{relative}: possible API key")
    if leaks:
        errors.append(f"Tracked text may leak local or secret values: {leaks}")
    if not forbidden and not leaks:
        passes.append(f"Git index confidentiality scan passed for {len(tracked)} files")


def _validate_predictions(frame: pd.DataFrame, errors: list[str], passes: list[str]) -> None:
    if list(frame.columns) != OUTPUT_COLUMNS:
        errors.append(
            "Prediction columns/order differ from the provisional output contract: "
            f"{list(frame.columns)}"
        )
        return
    if frame.empty:
        errors.append("Prediction output is empty")
        return

    expected_levels = {"campaign", "campaign_type", "channel", "overall"}
    if set(frame["forecast_level"]) != expected_levels:
        errors.append("Prediction output does not contain all four forecast levels")
    if set(frame["horizon_days"].astype(int)) != set(DEFAULT_HORIZONS):
        errors.append("Prediction output does not contain exactly 30/60/90-day horizons")

    key = [
        "forecast_level",
        "channel",
        "campaign_type",
        "campaign_id",
        "horizon_days",
    ]
    if frame.duplicated(key).any():
        errors.append("Prediction output has duplicate identity keys")

    numeric = [
        "budget",
        "revenue_p10",
        "revenue_p50",
        "revenue_p90",
        "roas_p10",
        "roas_p50",
        "roas_p90",
        "target_roas",
        "probability_target",
        "data_points",
    ]
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        errors.append("Prediction output contains non-finite numeric values")
    if (values.drop(columns=["probability_target"]) < 0).any().any():
        errors.append("Prediction output contains negative numeric values")
    if not frame["probability_target"].between(0, 1).all():
        errors.append("probability_target must be between zero and one")
    if not (
        (frame["revenue_p10"] <= frame["revenue_p50"])
        & (frame["revenue_p50"] <= frame["revenue_p90"])
        & (frame["roas_p10"] <= frame["roas_p50"])
        & (frame["roas_p50"] <= frame["roas_p90"])
    ).all():
        errors.append("Prediction quantiles are not ordered")

    positive = frame["budget"] > 0
    ratio_error = (
        frame.loc[positive, "roas_p50"]
        - frame.loc[positive, "revenue_p50"] / frame.loc[positive, "budget"]
    ).abs()
    if (ratio_error > 0.002).any():
        errors.append("ROAS and revenue/budget are inconsistent")

    for horizon in DEFAULT_HORIZONS:
        selected = frame[frame["horizon_days"] == horizon]
        campaign_budget = selected.loc[selected["forecast_level"] == "campaign", "budget"].sum()
        type_budget = selected.loc[selected["forecast_level"] == "campaign_type", "budget"].sum()
        channel_budget = selected.loc[selected["forecast_level"] == "channel", "budget"].sum()
        overall_budget = selected.loc[selected["forecast_level"] == "overall", "budget"].iloc[0]
        if not np.allclose(
            [campaign_budget, type_budget, channel_budget],
            overall_budget,
            rtol=0,
            atol=0.01,
        ):
            errors.append(f"Budget hierarchy does not reconcile at {horizon} days")
    if not errors:
        passes.append(f"prediction contract passed for {len(frame)} rows")


def _smoke_test(root: Path, errors: list[str], passes: list[str]) -> None:
    protected = [
        root / "pickle" / "model.pkl",
        *sorted((root / "data").glob("*.csv")),
        *sorted((root / "src").glob("*.py")),
    ]
    before = {str(path): _sha256(path) for path in protected}
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "*",
        }
    )
    with tempfile.TemporaryDirectory(prefix="range_submission_") as directory:
        work = Path(directory) / "paths with spaces"
        work.mkdir(parents=True)
        first = work / "first predictions.csv"
        second = work / "second predictions.csv"
        first.write_text("SENTINEL", encoding="utf-8")
        bash = _find_usable_bash()
        for output in (first, second):
            if bash:
                command = [
                    bash,
                    str(root / "run.sh"),
                    str(root / "data"),
                    str(root / "pickle" / "model.pkl"),
                    str(output),
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "src.predict",
                    "--data-dir",
                    str(root / "data"),
                    "--model",
                    str(root / "pickle" / "model.pkl"),
                    "--output",
                    str(output),
                ]
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            if result.returncode:
                errors.append(
                    f"Offline prediction smoke test failed ({result.returncode}): "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
                return
            if not output.is_file() or output.stat().st_size == 0:
                errors.append("Offline prediction smoke test did not create a non-empty output")
                return
        if first.read_text(encoding="utf-8").startswith("SENTINEL"):
            errors.append("Prediction runner appended to output instead of replacing it")
        if _sha256(first) != _sha256(second):
            errors.append("Two identical prediction runs produced different CSV hashes")
        frame = pd.read_csv(first)
        _validate_predictions(frame, errors, passes)

    after = {str(path): _sha256(path) for path in protected}
    if before != after:
        errors.append("Prediction smoke test modified an input, model, or source file")
    else:
        runner = "run.sh" if _find_usable_bash() else "prediction CLI fallback"
        passes.append(
            f"offline {runner} is deterministic and leaves protected files unchanged"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    _check_required_files(root, errors, passes)
    _check_requirements(root, errors, passes)
    _check_model(root, errors, passes)
    _check_run_scripts(root, errors, warnings, passes)
    _check_git_index(root, errors, warnings, passes)
    if not args.skip_smoke:
        _smoke_test(root, errors, passes)

    for message in passes:
        print(f"[PASS] {message}")
    for message in warnings:
        print(f"[WARN] {message}")
    for message in errors:
        print(f"[FAIL] {message}")
    if errors:
        raise SystemExit(1)
    print(f"Submission validation passed with {len(passes)} checks.")


if __name__ == "__main__":
    main()
