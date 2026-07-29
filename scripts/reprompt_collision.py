"""Build a re-prompt batch for plans that failed the geometric checks.

Two kinds of failure are fed back to the model, each as a follow-up turn on the
conversation that produced the plan:

* colliding part pairs (``failures.json``), with a hint naming the part that
  appears in the most collisions;
* parts left floating (``float.json``), i.e. with no overlap to connect to.

Runtime settings default to ``config/reprompt_collision_config.json`` and may be
overridden on the command line.

Usage:
    python reprompt_collision.py
    python reprompt_collision.py --obj-type chair --output-jsonl OUT.jsonl
"""
import argparse

import numpy as np

from batch_utils import (
    TaskCollector,
    build_conversation,
    format_paths,
    load_frame,
    load_json,
    user_message,
    write_tasks,
)
from config_loader import load_config

COLLISION_PROMPT = """The proposed plan generated a craft with the following description: {description}.
    It failed to generate a valid craft because of the following collision pairs: {pairs}.
    First, analyze if there are any errors in the relative placement of the parts. Consider make adjustments to the position and alignment accordingly.
    Then, if the relative placement is correct, consider that there may not be enough clearance for the part. Analyze it, and propose to modify the object correspondence or other adjustments.
    Reply with the full json formatted plan with corrections that address this issue. Do not output anything else. Do not hesitate to make large sections of the plan.
    """

COLLISION_HINT = """Suggestion: {part} may have and incorrect "align" or "to_face" parameter in their connections. Refer to the template and the relative position between parts. IF you think this part is not the issue, verify the other parts connected to it."""

FLOAT_PROMPT = """The proposed plan generated a craft with the following description: {description}.
    It failed to generate a valid craft the following parts are not connected: {pairs}.
    First, analyze if there are any errors in the relative placement of the parts. Consider make adjustments to the position and alignment accordingly.
    Then, if the relative placement is correct, consider that there may not be enough clearance for the part. Analyze it, and propose to modify the object correspondence or other adjustments.
    Reply with the full json formatted plan with corrections that address this issue. Do not output anything else. Do not hesitate to make large sections of the plan.
    """


def most_frequent_part(pairs):
    """The part appearing in the most failing pairs, and whether any repeats.

    A part that shows up in several collisions is usually the misplaced one, so
    it is worth naming in the feedback. When every part appears once there is
    nothing to single out.
    """
    flattened = [part for pair in pairs for part in pair]
    parts = list(set(flattened))
    counts = [flattened.count(part) for part in parts]
    if not parts:
        return None, False
    return parts[int(np.argmax(counts))], any(count > 1 for count in counts)


def add_feedback_tasks(collector, config, prompt_json, answer_json, failures, descriptions,
                       template, stage, with_hint=False):
    """Queue one re-prompt per failing id, quoting its failure back to the model."""
    for custom_id, pairs in failures.items():
        messages = build_conversation(prompt_json, custom_id, answer_json)
        if messages is None:
            continue

        suggested, repeated = most_frequent_part(pairs)
        print(custom_id, pairs)

        user_prompt = template.format(description=descriptions[custom_id], pairs=pairs)
        if with_hint and repeated:
            user_prompt += COLLISION_HINT.format(part=suggested)

        messages.append(user_message(user_prompt))
        collector.add(custom_id, messages, config["model"], config["reasoning_effort"], stage)


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-type", default=defaults["obj_type"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--reasoning-effort", default=defaults["reasoning_effort"])
    parser.add_argument("--prompt-jsonl", default=None)
    parser.add_argument("--answer-jsonl", default=None)
    parser.add_argument("--collision-json", default=None)
    parser.add_argument("--float-json", default=None)
    parser.add_argument("--description-json", default=None)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def main():
    defaults = load_config("reprompt_collision_config.json")
    args = parse_args(defaults)

    config = format_paths(defaults, obj_type=args.obj_type, model=args.model)
    config["obj_type"] = args.obj_type
    config["model"] = args.model
    config["reasoning_effort"] = args.reasoning_effort
    for key in ("prompt_jsonl", "answer_jsonl", "collision_json", "float_json",
                "description_json", "output_jsonl"):
        override = getattr(args, key)
        if override is not None:
            config[key] = override

    prompt_json = load_frame(config["prompt_jsonl"])
    answer_json = load_frame(config["answer_jsonl"])
    descriptions = load_json(config["description_json"])
    collisions = load_json(config["collision_json"], default={})
    floating = load_json(config["float_json"], default={})

    collector = TaskCollector()
    add_feedback_tasks(collector, config, prompt_json, answer_json, collisions, descriptions,
                       COLLISION_PROMPT, stage="collision", with_hint=True)
    add_feedback_tasks(collector, config, prompt_json, answer_json, floating, descriptions,
                       FLOAT_PROMPT, stage="float")

    write_tasks(config["output_jsonl"], collector.tasks)


if __name__ == "__main__":
    main()
