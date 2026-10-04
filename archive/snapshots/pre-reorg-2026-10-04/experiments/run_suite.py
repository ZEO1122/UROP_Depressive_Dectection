"""Run three local engineering pilots, keeping independent clinical evaluation pending."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .prepare import prepare, require_private_output
from .report import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, default=Path("Data/DAIC-WOZ"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/experiments/2026-09-27"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = require_private_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prepare(args.zip_dir, output)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(__file__).parent.glob("*.py"))}
    execution: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "train_dev_exploratory_only_not_preregistered_clinical_evaluation",
        "source_hashes": sources, "commands": [], "test_opened": False,
        "human_annotations": "not available; primary efficacy endpoints not evaluated",
    }
    for name in ("a", "b", "c"):
        command = [sys.executable, "-m", f"experiments.run_{name}",
                   "--data-dir", str(output), "--output-dir", str(output / f"results_{name}")]
        if name == "b":
            command.extend(["--zip-dir", str(args.zip_dir), "--workers", str(args.workers)])
        print(f"Executing experiment {name.upper()} (local diagnostic only)", flush=True)
        with (output / f"run_{name}.log").open("w") as stream:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        execution["commands"].append({"args": command, "returncode": result.returncode})
        (output / "execution.json").write_text(json.dumps(execution, indent=2))
        if result.returncode:
            raise RuntimeError(f"Experiment {name} failed. Inspect private run_{name}.log")
    execution["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "execution.json").write_text(json.dumps(execution, indent=2))
    summarize(output)
    print(f"Completed all three technical pilots: {output / 'results_summary_ko.md'}")


if __name__ == "__main__":
    main()
