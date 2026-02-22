from __future__ import annotations

import random

READING_PASSAGES = [
    "Read naturally: The human eye focuses using tiny muscles. Long screen sessions can reduce blink rate.",
    "Read naturally: The 20-20-20 rule suggests looking 20 feet away for 20 seconds every 20 minutes.",
    "Read naturally: Dry eyes often happen when we stare. Blinking spreads tears evenly across the eye surface.",
    "Read naturally: Good lighting reduces squinting. Avoid glare from windows or bright lights behind you.",
]


def get_focus_task(elapsed_sec: float):
    idx = int(elapsed_sec // 12) % len(READING_PASSAGES)
    title = "Focus Task (Read naturally)"
    body = READING_PASSAGES[idx]
    return title, body


def get_micro_prompt(elapsed_sec: float):
    prompts = [
        "Keep reading normally.",
        "Try not to over-think blinking—just be natural.",
        "Relax your shoulders and jaw.",
        "Maintain a comfortable distance from the screen.",
    ]
    return prompts[int(elapsed_sec // 8) % len(prompts)]


def simple_attention_question(seed: int = 1):
    random.seed(seed)
    target = random.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"))
    count = random.randint(4, 9)
    return target, count