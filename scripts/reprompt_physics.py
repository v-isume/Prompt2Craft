"""Build a re-prompt batch for plans that assembled but failed the physics test.

The simulator's verdict (``physics.jsonl``, written by ``check_physics.py``) is
quoted back to the model, which is asked for a fresh plan rather than a patch --
a craft that does not roll or does not hold together usually needs a different
mechanism, not a nudged alignment.

Two rounds are covered in one batch: crafts that failed physics on the first
attempt, and crafts that failed again after the collision re-prompt. An id
appearing in both is only re-prompted once, from its most recent conversation.

Runtime settings default to ``config/reprompt_physics_config.json`` and may be
overridden on the command line.

Usage:
    python reprompt_physics.py
    python reprompt_physics.py --obj-type truck --output-jsonl OUT.jsonl
"""
import argparse

from batch_utils import (
    TaskCollector,
    build_conversation,
    format_paths,
    load_frame,
    load_jsonl,
    user_message,
    write_tasks,
)
from config_loader import load_config

PHYSICS_PROMPT = """This craft plan is not valid because of the following: {reason}.
    Create a new craft assembly plan, from scratch, while avoiding the errors from the previous plan.
    """


def failure_reason(record, custom_id):
    """The simulator's verdict for ``custom_id``: failure reason plus details."""
    entries = record[custom_id]
    return entries[0]["failure_reason"] + entries[1]["additional_info"]


def add_physics_tasks(collector, config, prompt_json, answer_json, physics_records, stage):
    """Queue one re-prompt per craft the simulator rejected."""
    for record in physics_records:
        custom_id = list(record.keys())[0]
        messages = build_conversation(prompt_json, custom_id, answer_json)
        if messages is None:
            continue

        messages.append(user_message(
            PHYSICS_PROMPT.format(reason=failure_reason(record, custom_id))
        ))
        collector.add(custom_id, messages, config["model"], config["reasoning_effort"], stage)


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-type", default=defaults["obj_type"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--reasoning-effort", default=defaults["reasoning_effort"])
    parser.add_argument("--prompt-jsonl", default=None)
    parser.add_argument("--answer-jsonl", default=None)
    parser.add_argument("--physics-jsonl", default=None)
    parser.add_argument("--retry-prompt-jsonl", default=None)
    parser.add_argument("--retry-answer-jsonl", default=None)
    parser.add_argument("--retry-physics-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def main():
    defaults = load_config("reprompt_physics_config.json")
    args = parse_args(defaults)

    config = format_paths(defaults, obj_type=args.obj_type, model=args.model)
    config["obj_type"] = args.obj_type
    config["model"] = args.model
    config["reasoning_effort"] = args.reasoning_effort
    for key in ("prompt_jsonl", "answer_jsonl", "physics_jsonl", "retry_prompt_jsonl",
                "retry_answer_jsonl", "retry_physics_jsonl", "output_jsonl"):
        override = getattr(args, key)
        if override is not None:
            config[key] = override

    collector = TaskCollector()

    # First round: crafts that failed physics straight after the initial prompt.
    add_physics_tasks(
        collector, config,
        load_frame(config["prompt_jsonl"]),
        load_frame(config["answer_jsonl"]),
        load_jsonl(config["physics_jsonl"], default=[]),
        stage="first prompt",
    )

    # Second round: crafts that failed physics again after the collision re-prompt,
    # continued from that re-prompt's conversation.
    add_physics_tasks(
        collector, config,
        load_frame(config["retry_prompt_jsonl"]),
        load_frame(config["retry_answer_jsonl"]),
        load_jsonl(config["retry_physics_jsonl"], default=[]),
        stage="re-prompt",
    )

    write_tasks(config["output_jsonl"], collector.tasks)


if __name__ == "__main__":
    main()
