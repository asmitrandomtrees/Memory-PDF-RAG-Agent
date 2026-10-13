"""Fine-tuning pipeline for the conversational agent.

Usage:
  python scripts/finetune.py export [--conversations PATH] [--out DIR]
  python scripts/finetune.py submit [--train FILE] [--validation FILE]
                                    [--model NAME] [--suffix S] [--epochs N]
  python scripts/finetune.py status JOB_ID
  python scripts/finetune.py cancel JOB_ID
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings
from app.finetuning import azure_job
from app.finetuning.dataset import FineTuneExportConfig, export_dataset


def cmd_export(args: argparse.Namespace) -> None:
    s = get_settings()
    config = FineTuneExportConfig(
        max_history_messages=args.history,
        validation_ratio=args.validation_ratio,
        include_insufficient_evidence=args.include_insufficient,
        seed=args.seed,
    )
    result = export_dataset(
        args.conversations or s.conversation_data_path,
        args.out or s.finetune_data_path,
        config,
    )
    print(f"Training examples:   {result.training_count} -> {result.training_path}")
    if result.validation_path:
        print(f"Validation examples: {result.validation_count} -> {result.validation_path}")
    print(f"Skipped replies:     {result.skipped_count}")
    if result.training_count < 10:
        print("Warning: Azure OpenAI requires at least 10 training examples.")


def cmd_submit(args: argparse.Namespace) -> None:
    s = get_settings()
    data_dir = Path(s.finetune_data_path)
    train = Path(args.train or data_dir / "train.jsonl")
    validation = Path(args.validation) if args.validation else data_dir / "validation.jsonl"

    client = azure_job.create_client(s)
    training_file_id = azure_job.upload_training_file(client, train)
    print(f"Uploaded {train} -> {training_file_id}")
    validation_file_id = None
    if validation.is_file():
        validation_file_id = azure_job.upload_training_file(client, validation)
        print(f"Uploaded {validation} -> {validation_file_id}")

    job = azure_job.create_job(
        client,
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
        base_model=args.model or s.finetune_base_model,
        suffix=args.suffix,
        n_epochs=args.epochs,
    )
    print(f"Created fine-tuning job {job.id} (status: {job.status})")
    print(f"Check progress with: python scripts/finetune.py status {job.id}")


def cmd_status(args: argparse.Namespace) -> None:
    client = azure_job.create_client()
    job = azure_job.get_job(client, args.job_id)
    print(f"Job {job.id}: {job.status}")
    if job.fine_tuned_model:
        print(f"Fine-tuned model: {job.fine_tuned_model}")
        print(
            "Deploy it in Azure AI Foundry, then set OPENAI_DEPLOYMENT "
            "to the new deployment name."
        )
    if getattr(job, "error", None) and job.error.message:
        print(f"Error: {job.error.message}")
    for event in reversed(azure_job.list_job_events(client, args.job_id)):
        print(f"  [{event.created_at}] {event.message}")


def cmd_cancel(args: argparse.Namespace) -> None:
    job = azure_job.cancel_job(azure_job.create_client(), args.job_id)
    print(f"Job {job.id}: {job.status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="Build train/validation JSONL from stored conversations")
    p.add_argument("--conversations", help="Conversation store root (default: CONVERSATION_DATA_PATH)")
    p.add_argument("--out", help="Output directory (default: FINETUNE_DATA_PATH)")
    p.add_argument("--history", type=int, default=10, help="Prior messages per example")
    p.add_argument("--validation-ratio", type=float, default=0.1)
    p.add_argument("--include-insufficient", action="store_true",
                   help="Keep replies flagged insufficient_evidence")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("submit", help="Upload datasets and start an Azure OpenAI fine-tuning job")
    p.add_argument("--train", help="Training JSONL (default: FINETUNE_DATA_PATH/train.jsonl)")
    p.add_argument("--validation", help="Validation JSONL (default: FINETUNE_DATA_PATH/validation.jsonl if present)")
    p.add_argument("--model", help="Base model (default: FINETUNE_BASE_MODEL)")
    p.add_argument("--suffix", help="Suffix for the fine-tuned model name")
    p.add_argument("--epochs", type=int, help="Number of training epochs")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="Show job status and recent events")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("cancel", help="Cancel a running job")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_cancel)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
