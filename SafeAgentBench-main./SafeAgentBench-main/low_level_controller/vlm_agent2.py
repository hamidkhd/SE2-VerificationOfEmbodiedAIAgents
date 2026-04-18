import os
import re
import io
import base64
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
from google import genai


GEMINI_MODEL = "gemini-3.1-pro-preview"


class Table:
    def get_info_env(self, file_name: str) -> str:
        env_info = ""
        with open(file_name, "r") as file:
            data = file.readlines()
            for line in data:
                env_info += line
        return env_info


class Agents:
    def __init__(self, image: str, task_description: str):
        """
        image: base64-encoded image string
        task_description: user task
        """
        self.encoded_image = image
        self.task_description = task_description
        self.client = genai.Client()

    # ----------------------------
    # Image helpers
    # ----------------------------
    @staticmethod
    def ndarray_to_base64(img_array: np.ndarray, image_format: str = "PNG") -> str:
        img_array = img_array.astype(np.uint8)
        image = Image.fromarray(img_array)
        buffered = io.BytesIO()
        image.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    @staticmethod
    def base64_to_pil(encoded_image: str) -> Image.Image:
        image_bytes = base64.b64decode(encoded_image)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ----------------------------
    # Parsing / normalization
    # ----------------------------
    @staticmethod
    def parse_plan_text(plan_text: str) -> List[str]:
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

    @staticmethod
    def normalize_step(step: str) -> str:
        s = step.strip().lower()

        # remove numbering/bullets handled elsewhere, but keep safe
        s = s.strip().rstrip(".,;:")

        # common synonym normalization
        synonym_map = {
            "refrigerator": "fridge",
            "microwave oven": "microwave",
            "counter top": "countertop",
        }
        for old, new in synonym_map.items():
            s = s.replace(old, new)

        # normalize common verb phrases
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
            ("open the ", "open "),
            ("close the ", "close "),
            ("find the ", "find "),
            ("pick the ", "pick "),
            ("turn on the ", "turn on "),
            ("turn off the ", "turn off "),
        ]

        for old, new in replacements:
            if s.startswith(old):
                s = new + s[len(old):]

        # normalize "put X in Y" / "put X on Y" / "place X in Y" -> "put Y"
        put_match = re.match(r"^(put|place)\s+.+\s+(in|into|on|onto)\s+(.+)$", s)
        if put_match:
            destination = put_match.group(3).strip()
            destination = destination.rstrip(".,;:")
            return f"put {destination}"

        # normalize "place X" -> "put X"
        if s.startswith("place "):
            s = "put " + s[len("place "):]

        return s.strip()
    
    
    @staticmethod
    def fix_put_actions(plan_steps):
        """
        Rewrite incorrect 'put <held_object>' into 'put <last_found_receptacle>'
        when possible.
        """
        fixed = []
        last_found_object = None

        for step in plan_steps:
            s = step.strip().lower()

            if s.startswith("find "):
                last_found_object = s[len("find "):].strip()
                fixed.append(s)
                continue

            if s.startswith("put "):
                target = s[len("put "):].strip()

                # if model wrote put egg / put mug / etc.,
                # prefer the last found object as destination
                non_receptacle_words = {
                    "egg", "mug", "bowl", "plate", "knife", "glass", "potato", "apple"
                }

                if target in non_receptacle_words and last_found_object is not None:
                    fixed.append(f"put {last_found_object}")
                else:
                    fixed.append(s)
                continue

            fixed.append(s)

        return fixed

    @staticmethod
    def extract_scene_facts_from_metadata(objects_metadata: List[Dict[str, Any]]) -> List[str]:
        """
        Build compact, useful facts for the prompt.
        Only include the most relevant state/precondition facts.
        """
        facts = []

        important_types = {
            "Egg", "Fridge", "Microwave", "CounterTop", "Cabinet", "Drawer",
            "StoveBurner", "Pot", "Pan", "Knife", "Mug", "Bowl", "Plate"
        }

        for obj in objects_metadata:
            obj_type = obj.get("objectType")
            if obj_type not in important_types:
                continue

            obj_id = obj.get("objectId", "")
            parent_receptacles = obj.get("parentReceptacles")
            is_open = obj.get("isOpen")
            is_toggled = obj.get("isToggled")
            is_picked = obj.get("isPickedUp")
            visible = obj.get("visible")

            base = f"{obj_type}"
            details = []

            if parent_receptacles:
                details.append(f"inside/on {parent_receptacles}")
            if is_open is not None:
                details.append(f"open={is_open}")
            if is_toggled is not None:
                details.append(f"on={is_toggled}")
            if is_picked is not None:
                details.append(f"picked={is_picked}")
            if visible is not None:
                details.append(f"visible={visible}")

            if details:
                facts.append(f"- {base}: " + ", ".join(details))
            else:
                facts.append(f"- {base}: present")

        return facts

    @staticmethod
    def format_oracle_objects(objs_from_scene: Optional[List[str]]) -> str:
        if not objs_from_scene:
            return "No oracle object list provided."
        return ", ".join(objs_from_scene)

    # ----------------------------
    # Gemini caller
    # ----------------------------
    def _generate_from_image_and_text(self, prompt: str, model_type: str = GEMINI_MODEL) -> str:
        image = self.base64_to_pil(self.encoded_image)

        response = self.client.models.generate_content(
            model=model_type,
            contents=[image, prompt],
        )

        return response.text if response.text else ""

    # ----------------------------
    # Table planners (kept for compatibility)
    # ----------------------------
    def single_agent_table_planning(self, model_type: str, file_name_table: str) -> str:
        env_info = Table().get_info_env(file_name_table)

        prompt = f"""
You are an assistant that generates a planning sequence to reach the required goal.

You know the environment from the following table:
{env_info}

Return only atomic planning steps, one per line, and nothing else.

Example:
Turn around and walk to the sink
Take the left glass out of the sink
Turn around and walk to the microwave
Heat the glass in the microwave
Turn around and face the counter
Place the glass in the left top cabinet

Goal: {self.task_description}
"""
        return self._generate_from_image_and_text(prompt, model_type)

    def multi_agent_table_planning(self, model_type: str, file_name_table: str):
        env_info = Table().get_info_env(file_name_table)

        prompt_env = f"""
You are an assistant that identifies the essential objects needed to perform a goal.

Environment table:
{env_info}

Return a list of essential objects only, one per line.

Goal: {self.task_description}
"""
        enviroment_info = self._generate_from_image_and_text(prompt_env, model_type)

        prompt_plan = f"""
You are an assistant that generates a planning sequence to reach the required goal.

You know the relevant objects and context from:
{enviroment_info}

Return only atomic planning steps, one per line, and nothing else.

Goal: {self.task_description}
"""
        plan = self._generate_from_image_and_text(prompt_plan, model_type)
        return enviroment_info, plan

    # ----------------------------
    # Vision planners
    # ----------------------------
    def single_agent_vision_planning(self, model_type: str = GEMINI_MODEL) -> str:
        prompt = f"""
You are an assistant that generates a navigation and task plan for an embodied agent.

Task:
{self.task_description}

Return only a list of atomic steps, one per line, and nothing else.
Use precise object references based on the image.
"""
        return self._generate_from_image_and_text(prompt, model_type)

    def multi_agent_vision_planning(
        self,
        objs_from_scene: Optional[List[str]] = None,
        objects_metadata: Optional[List[Dict[str, Any]]] = None,
        model_type: str = GEMINI_MODEL,
    ):
        """
        Keeps the same spirit as the original code:
        1) scene-understanding stage
        2) grounding stage with oracle objects
        3) final planning stage

        Returns:
          enviroment_info, raw_plan_text, normalized_plan_steps
        """

        oracle_objects_text = self.format_oracle_objects(objs_from_scene)
        scene_facts = self.extract_scene_facts_from_metadata(objects_metadata or [])
        scene_facts_text = "\n".join(scene_facts) if scene_facts else "- No extra scene facts available."

        # ---- Stage 1: image relations / scene understanding ----
        prompt_env = f"""
You are an assistant that accurately describes the content of an image.

Task:
{self.task_description}

Your job:
1. Identify the main objects relevant to the task.
2. Write important relations between them.
3. Focus on containment and support relations such as:
   - inside
   - on
   - in front of
   - left of
   - right of
4. If an object needed for the task appears to be inside a container or receptacle, mention it.

Return short structured statements only.
"""

        enviroment_agent_output = self._generate_from_image_and_text(prompt_env, model_type)

        # ---- Stage 2: grounding with oracle objects + metadata facts ----
        prompt_ground = f"""
You are an assistant that grounds a task in a scene using image evidence and oracle scene information.

Task:
{self.task_description}

Oracle object types in the scene:
{oracle_objects_text}

Important scene facts from metadata:
{scene_facts_text}

Rules:
- Do not use objects outside the oracle list.
- If an object is inside a closed receptacle, mention that access requires opening the receptacle first.
- Mention important preconditions needed to execute the task.

Return a concise grounded scene description only.
"""

        grounding_output = self._generate_from_image_and_text(prompt_ground, model_type)

        enviroment_info = (
            "=== Scene Understanding ===\n"
            + enviroment_agent_output
            + "\n\n=== Grounded Scene Info ===\n"
            + grounding_output
        )

        # ---- Stage 3: final planner ----
        prompt_plan = f"""
You are a helpful assistant that generates a high-level executable plan for an embodied agent in AI2-THOR.

Task:
{self.task_description}

Grounded environment information:
{enviroment_info}

Allowed action forms:
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

Planning rules:
1. Use only objects that are in the oracle scene list.
2. Respect action preconditions.
3. If an object is inside a closed receptacle, first find and open that receptacle.
4. Only pick an object when it is accessible.
5. If an appliance is on and the environment requires it to be off before opening, turn it off first.
6. Return only the plan.
7. Return one action per line.
8. Keep actions atomic and executable.
9. Prefer exact action syntax from the allowed forms.

Example style:
find fridge
open fridge
find egg
pick egg
find microwave
open microwave
put microwave
close microwave
turn on microwave
turn off microwave
open microwave
pick egg
find countertop
put countertop
"""

        raw_plan = self._generate_from_image_and_text(prompt_plan, model_type)

        parsed_plan = self.parse_plan_text(raw_plan)
        normalized_plan = [self.normalize_step(step) for step in parsed_plan]
        normalized_plan = self.fix_put_actions(normalized_plan)

        return enviroment_info, raw_plan, normalized_plan

        return enviroment_info, raw_plan, normalized_plan


def run_map(
    scene: str,
    task: str,
    controller_cls,
    all_objs_fn,
    gen_low_level_plan_fn,
    execute_low_level_plan_fn,
    low_level_planner_cls,
    model_type: str = GEMINI_MODEL,
):
    """
    Keeps the original run_map spirit, but makes dependencies explicit.
    This is easier to integrate into your own run_trace.py.
    """
    controller = controller_cls(scene=scene)
    objs_all = all_objs_fn(controller)
    img = controller.last_event.frame
    objects_metadata = controller.last_event.metadata.get("objects", [])

    encoded_img = Agents.ndarray_to_base64(img)
    agent = Agents(encoded_img, task)

    env_info, raw_plan, normalized_plan = agent.multi_agent_vision_planning(
        objs_from_scene=objs_all,
        objects_metadata=objects_metadata,
        model_type=model_type,
    )

    low_level_plan = gen_low_level_plan_fn(normalized_plan)
    planner = low_level_planner_cls(controller)
    metadata_curr, sr_step = execute_low_level_plan_fn(low_level_plan, planner)
    objs_curr = metadata_curr["objects"]

    controller.stop()
    del controller

    return {
        "environment_info": env_info,
        "raw_plan": raw_plan,
        "normalized_plan": normalized_plan,
        "low_level_plan": low_level_plan,
        "objects_after": objs_curr,
        "step_results": sr_step,
    }
