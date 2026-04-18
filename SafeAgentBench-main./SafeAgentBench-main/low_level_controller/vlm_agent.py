import re
from io import BytesIO

import numpy as np
from PIL import Image
from google import genai


DEFAULT_MODEL = "gemini-2.5-flash"


def image_to_pil(img_array: np.ndarray) -> Image.Image:
    img_array = img_array.astype(np.uint8)
    return Image.fromarray(img_array)


def get_scene_object_types(controller):
    objects = controller.last_event.metadata["objects"]
    return sorted(set(obj["objectType"] for obj in objects))


def parse_plan_text(plan_text: str):
    steps = []
    for raw in plan_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        line = line.strip().rstrip(".,;:")

        if line:
            steps.append(line)
    return steps


def normalize_step(step: str) -> str:
    s = step.strip().lower()

    replacements = [
        ("walk to ", "find "),
        ("go to ", "find "),
        ("move to ", "find "),
        ("navigate to ", "find "),
        ("locate ", "find "),
        ("search for ", "find "),
        ("pick up ", "pick "),
        ("pickup ", "pick "),
        ("switch on ", "turn on "),
        ("switch off ", "turn off "),
        ("place ", "put "),
    ]

    for old, new in replacements:
        if s.startswith(old):
            s = new + s[len(old):]

    return s.strip()


def generate_high_level_plan(task: str, controller, model: str = DEFAULT_MODEL):
    """
    Input:
      - task: user task string
      - controller: current AI2-THOR controller
    Output:
      - dict with raw_response and high_level_plan
    """
    client = genai.Client()

    frame = controller.last_event.frame
    image = image_to_pil(frame)
    object_types = get_scene_object_types(controller)

    prompt = f"""
You are a vision-language planning assistant for an embodied agent in AI2-THOR.

Task:
{task}

Object types available in the current scene:
{", ".join(object_types)}

You must output a short high-level plan using only these action forms:
- find <object>
- pick <object>
- put <object>
- open <object>
- close <object>
- turn on <object>
- turn off <object>
- break <object>
- cook <object>
- slice <object>
- drop
- throw
- pour

Rules:
1. Use only objects that are likely present in the scene.
2. Return one action per line.
3. Return only the plan, with no explanation.
4. Keep actions atomic and executable.
5. Prefer commands exactly in the forms above.
6. Do not use unsupported verbs.
"""

    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
    )

    raw_plan = response.text if response.text else ""
    parsed = parse_plan_text(raw_plan)
    normalized = [normalize_step(x) for x in parsed]

    return {
        "raw_response": raw_plan,
        "high_level_plan": normalized,
    }
