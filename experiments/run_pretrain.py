from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.runtime import enable_torch_import, ensure_preflight, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain conditioned generators on synthetic formula recovery.")
    parser.add_argument("--data-config", type=Path, default=Path("configs/us_equities_smoke.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--backtest-config", type=Path, default=Path("configs/backtest.yaml"))
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiments/us_equities_anchor.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-name", type=str, default="pretrain_us_equities")
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--length", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight = ensure_preflight("train")
    enable_torch_import()

    from knowledge_guided_symbolic_alpha.training import GeneratorPretrainer, SyntheticRecoveryDatasetBuilder

    from experiments.common import (
        build_sequence_generator,
        ensure_output_dirs,
        load_yaml,
        training_seed,
        write_json,
    )

    training_config = load_yaml(args.training_config)
    pretraining_config = training_config.get("training", {}).get("pretraining", {})
    examples = int(args.examples or pretraining_config.get("synthetic_examples", 128))
    epochs = int(args.epochs or pretraining_config.get("synthetic_epochs", 3))
    length = int(args.length or pretraining_config.get("synthetic_length", 96))
    seed = training_seed(training_config)

    output_dirs = ensure_output_dirs(args.output_root, args.run_name)
    manifest_path = write_run_manifest(
        output_dirs,
        script_name="experiments/run_pretrain.py",
        profile="train",
        preflight=preflight.to_dict(),
        config_paths={
            "data_config": str(args.data_config),
            "training_config": str(args.training_config),
            "backtest_config": str(args.backtest_config),
            "experiment_config": str(args.experiment_config),
        },
        dataset_name="us_equities",
        seed=seed,
        extra={"examples": examples, "epochs": epochs, "length": length},
    )

    builder = SyntheticRecoveryDatasetBuilder(seed=seed, length=length)
    dataset = builder.build(examples)
    generator = build_sequence_generator(training_config, seed=seed)
    pretrainer = GeneratorPretrainer(seed=seed)
    summary = pretrainer.fit(generator, dataset, epochs=epochs)

    checkpoint_path = args.checkpoint or Path(
        pretraining_config.get("checkpoint", output_dirs["checkpoints"] / "synthetic_prior_transformer.pt")
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(generator, "save_checkpoint"):
        raise RuntimeError("Configured generator does not support checkpoint saving.")
    generator.save_checkpoint(checkpoint_path)

    report_path = output_dirs["reports"] / "synthetic_pretraining.json"
    payload = {
        "examples": summary.examples,
        "epochs": summary.epochs,
        "exact_recovery_rate": summary.exact_recovery_rate,
        "first_token_accuracy": summary.first_token_accuracy,
        "mean_supervised_loss": summary.mean_supervised_loss,
        "checkpoint": str(checkpoint_path),
        "generator": training_config.get("training", {}).get("generator", "transformer"),
        "seed": seed,
        "length": length,
        "manifest": str(manifest_path),
    }
    write_json(report_path, payload)

    print(f"examples={summary.examples}")
    print(f"epochs={summary.epochs}")
    print(f"exact_recovery_rate={summary.exact_recovery_rate:.6f}")
    print(f"first_token_accuracy={summary.first_token_accuracy:.6f}")
    print(f"mean_supervised_loss={summary.mean_supervised_loss:.6f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
