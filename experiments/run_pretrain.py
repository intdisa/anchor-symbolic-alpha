from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.training import GeneratorPretrainer, SyntheticRecoveryDatasetBuilder

from experiments.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRAINING_CONFIG,
    build_sequence_generator,
    ensure_output_dirs,
    load_yaml,
    training_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain conditioned generators on synthetic formula recovery.")
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--length", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_config = load_yaml(args.training_config)
    pretraining_config = training_config.get("training", {}).get("pretraining", {})
    examples = int(args.examples or pretraining_config.get("synthetic_examples", 128))
    epochs = int(args.epochs or pretraining_config.get("synthetic_epochs", 3))
    length = int(args.length or pretraining_config.get("synthetic_length", 96))
    seed = training_seed(training_config)

    builder = SyntheticRecoveryDatasetBuilder(seed=seed, length=length)
    dataset = builder.build(examples)
    generator = build_sequence_generator(training_config, seed=seed)
    pretrainer = GeneratorPretrainer(seed=seed)
    summary = pretrainer.fit(generator, dataset, epochs=epochs)

    output_dirs = ensure_output_dirs(args.output_root)
    checkpoint_path = args.checkpoint or Path(pretraining_config.get("checkpoint", output_dirs["checkpoints"] / "synthetic_prior_transformer.pt"))
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
    }
    write_json(report_path, payload)

    print(f"examples={summary.examples}")
    print(f"epochs={summary.epochs}")
    print(f"exact_recovery_rate={summary.exact_recovery_rate:.6f}")
    print(f"first_token_accuracy={summary.first_token_accuracy:.6f}")
    print(f"mean_supervised_loss={summary.mean_supervised_loss:.6f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
