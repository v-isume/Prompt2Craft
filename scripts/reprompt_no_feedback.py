"""Build a no-feedback re-prompt batch: retry every failure with the same prompt.

This is the ablation baseline for the ``reprompt_collision`` / ``reprompt_physics``
scripts. Instead of telling the model what went wrong, it simply re-issues the
original prompt for each id that failed anywhere in the pipeline, so any
improvement can be attributed to the feedback rather than to a second sample.

Failures are collected from all four sources a round can produce -- format
errors, collisions, floating parts and physics -- and an id that appears in
several of them is only re-prompted once.

Runtime settings default to ``config/reprompt_no_feedback_config.json`` and may
be overridden on the command line.

Usage:
    python reprompt_no_feedback.py
    python reprompt_no_feedback.py --obj-type truck --output-jsonl OUT.jsonl
"""
import argparse

from batch_utils import (
    TaskCollector,
    build_conversation,
    format_paths,
    frame_ids,
    load_frame,
    load_json,
    load_jsonl,
    write_tasks,
)
from config_loader import load_config


def add_retry_tasks(collector, config, prompt_json, custom_ids, stage):
    """Queue an unchanged re-run of the original prompt for each id."""
    for custom_id in custom_ids:
        messages = build_conversation(prompt_json, custom_id)
        if messages is None:
            continue
        collector.add(custom_id, messages, config["model"], config["reasoning_effort"], stage)


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-type", default=defaults["obj_type"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--reasoning-effort", default=defaults["reasoning_effort"])
    parser.add_argument("--prompt-jsonl", default=None)
    parser.add_argument("--error-jsonl", default=None)
    parser.add_argument("--collision-json", default=None)
    parser.add_argument("--float-json", default=None)
    parser.add_argument("--physics-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def main():
    defaults = load_config("reprompt_no_feedback_config.json")
    args = parse_args(defaults)

    config = format_paths(defaults, obj_type=args.obj_type, model=args.model)
    config["obj_type"] = args.obj_type
    config["model"] = args.model
    config["reasoning_effort"] = args.reasoning_effort
    for key in ("prompt_jsonl", "error_jsonl", "collision_json", "float_json",
                "physics_jsonl", "output_jsonl"):
        override = getattr(args, key)
        if override is not None:
            config[key] = override

    prompt_json = load_frame(config["prompt_jsonl"])
    error_json = load_frame(config["error_jsonl"])
    collisions = load_json(config["collision_json"], default={})
    floating = load_json(config["float_json"], default={})
    physics_records = load_jsonl(config["physics_jsonl"], default=[])

    # Each physics record is a single-key dict {custom_id: [reasons]}.
    physics_ids = [list(record.keys())[0] for record in physics_records]

    collector = TaskCollector()
    add_retry_tasks(collector, config, prompt_json, frame_ids(error_json), "error")
    add_retry_tasks(collector, config, prompt_json, list(collisions.keys()), "collision")
    add_retry_tasks(collector, config, prompt_json, list(floating.keys()), "float")
    add_retry_tasks(collector, config, prompt_json, physics_ids, "physics")

    write_tasks(config["output_jsonl"], collector.tasks)


if __name__ == "__main__":
    main()
