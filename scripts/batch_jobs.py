"""Submitting, tracking and collecting OpenAI batch jobs.

A batch is tagged with what it is -- object type, model, pipeline round -- when
it is submitted, so it can be found again later without anyone writing an id
down. ``client.batches.list()`` returns the tags with each batch, which is what
lets ``retrieve_batch.py`` locate the job ``upload_batch.py`` created.

The answer file is named after the batch's input file plus ``_answer``, the
convention the ``reprompt_*`` configs already expect, so a retrieved batch lands
where the next stage looks for it.
"""
import os
import time

# Statuses a batch can still move out of; anything else is final.
PENDING_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}

# Tags written on submit and matched on retrieve. The API stores metadata values
# as strings, so everything is stringified going in.
TAG_KEYS = ("obj_type", "model", "round", "input_file")


def make_tags(**fields):
    """Batch metadata: the tags a job can later be found by."""
    return {key: str(fields[key]) for key in TAG_KEYS if fields.get(key) is not None}


def submit(client, input_path, endpoint="/v1/chat/completions", completion_window="24h", tags=None):
    """Upload a task JSONL and start a batch over it."""
    with open(input_path, "rb") as handle:
        batch_file = client.files.create(file=handle, purpose="batch")

    return client.batches.create(
        input_file_id=batch_file.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=tags or {},
    )


def find_batches(client, limit=100, **filters):
    """Batches whose tags match every filter given, newest first."""
    wanted = {key: str(value) for key, value in filters.items() if value is not None}
    matches = []
    for batch in client.batches.list(limit=limit):
        tags = dict(batch.metadata or {})
        if all(tags.get(key) == value for key, value in wanted.items()):
            matches.append(batch)
    return matches


def resolve(client, batch_id=None, limit=100, **filters):
    """The batch to act on: an explicit id, or the newest one matching the tags."""
    if batch_id:
        return client.batches.retrieve(batch_id)

    matches = find_batches(client, limit=limit, **filters)
    if not matches:
        described = ", ".join(f"{k}={v}" for k, v in filters.items() if v is not None)
        raise LookupError(
            f"No batch found with {described or 'those tags'}. Pass --batch-id explicitly, "
            "or check that the job was submitted by upload_batch.py (older jobs carry no tags)."
        )
    return matches[0]


def describe(batch):
    """One line summarising a batch, for listings and progress output."""
    tags = dict(batch.metadata or {})
    label = "/".join(tags.get(key, "?") for key in ("obj_type", "model", "round"))
    counts = batch.request_counts
    progress = f"{counts.completed}/{counts.total} done, {counts.failed} failed" if counts else ""
    return f"{batch.id}  {batch.status:<12} {label:<28} {progress}"


def wait_for(client, batch_id, poll_seconds=60, timeout_seconds=None):
    """Poll until the batch reaches a final status, printing progress."""
    started = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  {describe(batch)}")
        if batch.status not in PENDING_STATUSES:
            return batch
        if timeout_seconds is not None and time.time() - started > timeout_seconds:
            print(f"Still {batch.status} after {timeout_seconds}s, giving up on waiting. "
                  f"Retrieve it later with: retrieve_batch.py --batch-id {batch.id}")
            return batch
        time.sleep(poll_seconds)


def answer_path(client, batch, output_dir=None):
    """Where this batch's answers belong: ``<input stem>_answer.jsonl``."""
    tags = dict(batch.metadata or {})
    input_name = tags.get("input_file")
    if not input_name:
        # Older or externally created batches carry no tags; ask the API instead.
        input_name = os.path.basename(client.files.retrieve(batch.input_file_id).filename)

    stem = os.path.splitext(os.path.basename(input_name))[0]
    folder = output_dir if output_dir is not None else os.path.dirname(input_name)
    return os.path.join(folder or ".", f"{stem}_answer.jsonl")


def download(client, batch, out_path):
    """Write the batch's answers, and its errors alongside them if there are any.

    Returns the paths written. A batch that failed outright has no output file,
    only errors, so the caller is told what actually landed.
    """
    written = []
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).content
        with open(out_path, "wb") as handle:
            handle.write(content)
        written.append(out_path)
        print(f"Wrote answers -> {out_path}")
    else:
        print(f"Batch {batch.id} produced no output file (status: {batch.status}).")

    if batch.error_file_id:
        error_path = out_path.replace("_answer.jsonl", "_errors.jsonl")
        if error_path == out_path:
            error_path = out_path + ".errors"
        content = client.files.content(batch.error_file_id).content
        with open(error_path, "wb") as handle:
            handle.write(content)
        written.append(error_path)
        print(f"Some requests failed, details -> {error_path}")

    return written
