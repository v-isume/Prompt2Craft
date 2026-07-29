"""Shared helpers for building and reading OpenAI batch jobs.

Every prompting script in this folder speaks the same two file formats:

* a *task* JSONL -- one ``/v1/chat/completions`` request per line, as accepted
  by the OpenAI Batch API. Written by ``batch_prompt.py`` and the ``reprompt_*``
  scripts, uploaded by ``upload_batch.py``.
* an *answer* JSONL -- one batch result per line, downloaded by
  ``retrieve_batch.py``.

The re-prompting scripts all follow the same shape: read the tasks and answers
of a previous round, look up the conversation for a ``custom_id``, append a
feedback message, and emit a new task JSONL. The helpers here cover that shape
so each script only has to express the feedback it wants to give.
"""
import copy
import json
import os

import pandas as pd

BATCH_ENDPOINT = "/v1/chat/completions"


def format_paths(config, **fields):
    """Expand ``{obj_type}``/``{model}`` placeholders in every string setting."""
    return {
        key: value.format(**fields) if isinstance(value, str) else value
        for key, value in config.items()
    }


def load_json(path, default=None):
    """Parsed JSON contents of ``path``.

    ``failures.json`` and ``float.json`` are not written by every pipeline
    round, so a caller can pass ``default`` to treat a missing file as empty
    instead of failing. The substitution is reported, never silent.
    """
    if default is not None and not os.path.isfile(path):
        print(f"WARNING: {path} not found, continuing with {default!r}")
        return default
    with open(path, mode="r") as file:
        return json.load(file)


def load_jsonl(path, default=None):
    """Records of a JSONL file as a list of dicts (see :func:`load_json`)."""
    if default is not None and not os.path.isfile(path):
        print(f"WARNING: {path} not found, continuing with {default!r}")
        return default
    with open(path, mode="r") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_frame(path):
    """A JSONL file as a DataFrame, the form the batch files are indexed by."""
    return pd.read_json(path, lines=True)


def frame_ids(frame, column="custom_id"):
    """The ids held by a JSONL frame; empty when the file had no records.

    An error file with nothing in it parses to a frame with no columns at all,
    so the column cannot simply be indexed.
    """
    if column not in frame:
        return []
    return list(frame[column])


def get_prompt_messages(prompt_json, custom_id):
    """Messages originally sent for ``custom_id``, or ``None`` if absent.

    The list is copied: callers append feedback turns to it, and mutating the
    DataFrame's own list would corrupt later lookups of the same row.
    """
    for i in range(len(prompt_json)):
        if prompt_json["custom_id"][i] == custom_id:
            return copy.deepcopy(prompt_json["body"][i]["messages"])
    return None


def get_answer_message(answer_json, custom_id):
    """Assistant reply returned for ``custom_id``, or ``None`` if absent."""
    for i in range(len(answer_json)):
        if answer_json["custom_id"][i] == custom_id:
            return copy.deepcopy(answer_json["response"][i]["body"]["choices"][0]["message"])
    return None


def build_conversation(prompt_json, custom_id, answer_json=None):
    """The prior conversation for ``custom_id``: prompt plus optional reply.

    Returns ``None`` when the id has no prompt, so the caller can skip it rather
    than silently re-prompting with whatever conversation came before it.
    """
    messages = get_prompt_messages(prompt_json, custom_id)
    if messages is None:
        print(f"WARNING: no prompt found for {custom_id}, skipping")
        return None
    if answer_json is not None:
        reply = get_answer_message(answer_json, custom_id)
        if reply is None:
            print(f"WARNING: no answer found for {custom_id}, skipping")
            return None
        messages.append(reply)
    return messages


def user_message(text):
    """A user turn in the content-list form the batch files use."""
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def make_task(custom_id, messages, model, reasoning_effort):
    """One line of a batch task JSONL."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
    }


def write_tasks(path, tasks):
    """Write ``tasks`` as a batch task JSONL, creating the parent folder."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as file:
        for task in tasks:
            file.write(json.dumps(task) + "\n")
    print(f"Wrote {len(tasks)} tasks -> {path}")


class TaskCollector:
    """Accumulates batch tasks, keeping one task per ``custom_id``.

    An id can fail several checks in the same round (a collision *and* a
    physics failure, say). The later stage wins, matching the order the caller
    adds them in, and the replacement is reported so a surprising overlap is
    visible in the log.
    """

    def __init__(self):
        self.tasks = []
        self.custom_ids = []

    def add(self, custom_id, messages, model, reasoning_effort, stage=""):
        if custom_id in self.custom_ids:
            index = self.custom_ids.index(custom_id)
            self.tasks.pop(index)
            self.custom_ids.pop(index)
            where = f" now in {stage}" if stage else ""
            print(f"Repeated id, replacing the earlier task{where}: {custom_id}")
        self.custom_ids.append(custom_id)
        self.tasks.append(make_task(custom_id, messages, model, reasoning_effort))

    def __len__(self):
        return len(self.tasks)
