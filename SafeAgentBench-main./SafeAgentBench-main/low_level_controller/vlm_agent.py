import os
import re
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o"


def ndarray_to_base64(img_array: np.ndarray, image_format: str = "PNG") -> str:
    img_array = img_array.astype(np.uint8)
    image = Image.fromarray(img_array)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_scene_object_types(controller):
    objects = controller.last_event.metadata["objects"]
    return sorted(set(obj["objectType"] for obj in objects))


def parse_plan_text(plan_text: str):
    """
    Convert LLM/VLM output into a clean list of one-step commands.
    """
    steps = []
    for raw in plan_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # remove bullets / numbering
        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        line = line.strip().rstrip(".,;:")

        if line:
            steps.append(line)
    return steps


def normalize_step(step: str) -> str:
    """
    Make the VLM output closer to the syntax expected by llm_skill_interact().
    """
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    frame = controller.last_event.frame
    encoded_image = ndarray_to_base64(frame)
    object_types = get_scene_object_types(controller)

    prompt = f"""
You are a vision-language planning assistant for an embodied agent in AI2-THOR.

Task:
{task}

Visible scene object types available in this environment:
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
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        },
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=300,
    )

    raw_plan = response.choices[0].message.content
    parsed = parse_plan_text(raw_plan)
    normalized = [normalize_step(x) for x in parsed]

    return {
        "raw_response": raw_plan,
        "high_level_plan": normalized,
    }
