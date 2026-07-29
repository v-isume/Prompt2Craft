"""Collect the answers from a batch submitted by ``upload_batch.py``.

The batch is found by the tags it was submitted with -- object type, model,
round -- so there is no id to copy. Pass ``--batch-id`` for a job that predates
the tagging, and ``--list`` to see what is on the account.

Answers are written to ``<input stem>_answer.jsonl``, which is the path the
``reprompt_*`` configs already read, and any per-request failures land next to
them in ``_errors.jsonl``.

Runtime settings default to ``config/batch_job_config.json`` and may be
overridden on the command line.

Usage:
    python retrieve_batch.py --obj-type truck
    python retrieve_batch.py --list
    python retrieve_batch.py --obj-type truck --wait
    python retrieve_batch.py --batch-id batch_681b20fa82548190962cfcaad9b844d9
"""
import argparse

from openai import OpenAI

import batch_jobs
from config_loader import load_config


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=None,
                        help="retrieve this exact batch instead of searching by tag")
    parser.add_argument("--obj-type", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--round", default=None,
                        help="which pipeline round to collect, e.g. initial, collision, physics")
    parser.add_argument("--list", action="store_true",
                        help="show recent batches and exit, without downloading")
    parser.add_argument("--wait", action="store_true",
                        help="poll until the batch finishes instead of reporting its status")
    parser.add_argument("--poll-seconds", type=int, default=defaults["poll_seconds"])
    parser.add_argument("--output-dir", default=defaults["output_dir"],
                        help="where to write answers (default: next to the input file)")
    parser.add_argument("--output-jsonl", default=None,
                        help="exact answer path, overriding the derived one")
    return parser.parse_args()


def main():
    defaults = load_config("batch_job_config.json")
    args = parse_args(defaults)
    client = OpenAI()

    filters = {"obj_type": args.obj_type, "model": args.model, "round": args.round}

    if args.list:
        matches = batch_jobs.find_batches(client, **filters)
        if not matches:
            print("No batches match.")
            return
        for batch in matches:
            print(batch_jobs.describe(batch))
        return

    batch = batch_jobs.resolve(client, batch_id=args.batch_id, **filters)
    print(batch_jobs.describe(batch))

    if batch.status in batch_jobs.PENDING_STATUSES:
        if not args.wait:
            print(f"\nThe batch is still {batch.status}; nothing to download yet. "
                  f"Re-run with --wait to poll until it finishes.")
            return
        print("\nWaiting...")
        batch = batch_jobs.wait_for(client, batch.id, poll_seconds=args.poll_seconds)

    if batch.status != "completed" and not batch.output_file_id:
        print(f"\nThe batch ended as {batch.status}, so there are no answers to collect.")
        if batch.errors:
            print(batch.errors)
        return

    out_path = args.output_jsonl or batch_jobs.answer_path(client, batch, args.output_dir)
    batch_jobs.download(client, batch, out_path)


if __name__ == "__main__":
    main()
