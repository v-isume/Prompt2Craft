"""Submit prompt batches to the OpenAI Batch API.

Each job is tagged with the object type, model and pipeline round it belongs to,
so ``retrieve_batch.py`` can find it later without anyone copying a batch id
around. Several object types can be submitted in one go.

With ``--wait`` the script stays up until the jobs finish and writes the answers
straight to the ``_answer.jsonl`` path the ``reprompt_*`` scripts read, making
the whole round a single command.

Runtime settings default to ``config/batch_job_config.json`` and may be
overridden on the command line.

Usage:
    python upload_batch.py --obj-type truck
    python upload_batch.py --obj-type truck --obj-type chair --round collision
    python upload_batch.py --obj-type truck --wait
"""
import argparse
import os

from openai import OpenAI

import batch_jobs
from config_loader import load_config


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-type", action="append", default=None,
                        help="object type to submit; repeat for several")
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--round", default=defaults["round"],
                        help="which pipeline round this is, e.g. initial, collision, physics")
    parser.add_argument("--input-jsonl", default=None,
                        help="task file to submit; overrides the configured pattern "
                             "(only valid with a single --obj-type)")
    parser.add_argument("--endpoint", default=defaults["endpoint"])
    parser.add_argument("--completion-window", default=defaults["completion_window"])
    parser.add_argument("--wait", action="store_true",
                        help="poll until the jobs finish, then download the answers")
    parser.add_argument("--poll-seconds", type=int, default=defaults["poll_seconds"])
    parser.add_argument("--output-dir", default=defaults["output_dir"],
                        help="where --wait writes answers (default: next to the input file)")
    return parser.parse_args()


def main():
    defaults = load_config("batch_job_config.json")
    args = parse_args(defaults)

    obj_types = args.obj_type or [defaults["obj_type"]]
    if args.input_jsonl and len(obj_types) > 1:
        raise SystemExit("--input-jsonl applies to one object type; pass a single --obj-type.")

    client = OpenAI()
    submitted = []

    for obj_type in obj_types:
        input_path = args.input_jsonl or defaults["input_jsonl"].format(
            model=args.model, obj_type=obj_type)
        if not os.path.isfile(input_path):
            raise SystemExit(
                f"No task file at {input_path} (paths are relative to the working directory).\n"
                f"Build it first with batch_prompt.py, adjust \"input_jsonl\" in "
                f"config/batch_job_config.json, or pass --input-jsonl."
            )

        tags = batch_jobs.make_tags(obj_type=obj_type, model=args.model, round=args.round,
                                    input_file=input_path)
        batch = batch_jobs.submit(client, input_path, endpoint=args.endpoint,
                                  completion_window=args.completion_window, tags=tags)
        submitted.append(batch)
        print(f"Submitted {input_path}")
        print(f"  {batch_jobs.describe(batch)}")

    if not args.wait:
        print("\nCollect them later with:")
        for batch, obj_type in zip(submitted, obj_types):
            print(f"  python retrieve_batch.py --obj-type {obj_type} --round {args.round}")
        return

    print("\nWaiting for the jobs to finish...")
    for batch in submitted:
        finished = batch_jobs.wait_for(client, batch.id, poll_seconds=args.poll_seconds)
        if finished.status == "completed" or finished.output_file_id:
            batch_jobs.download(client, finished,
                                batch_jobs.answer_path(client, finished, args.output_dir))
        else:
            print(f"  {finished.id} ended as {finished.status}, nothing to download.")


if __name__ == "__main__":
    main()
