import os
import re
import io
import base64
from typing import List, Dict, Any, Optional, Set
import numpy as np
from PIL import Image
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account


GEMINI_MODEL = "gemini-2.5-pro"

KEY_PATH = "cred.json"
PROJECT_ID = "robust-slice-493713-e1"
LOCATION = "us-central1"


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

        creds = service_account.Credentials.from_service_account_file(KEY_PATH)

        vertexai.init(
            project=PROJECT_ID,
            location=LOCATION,
            credentials=creds
        )

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

    
    @staticmethod
    def save_model_input_image(
        encoded_image: str,
        out_dir: str = "model_input_images",
        filename: str = "model_input.png",
    ) -> str:
        """
        Saves the exact PNG bytes that are sent to the model prompt.
        Returns the saved file path.
        """
        os.makedirs(out_dir, exist_ok=True)

        image = Agents.base64_to_pil(encoded_image)

        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        out_path = os.path.join(out_dir, filename)
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        return out_path
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
        s = s.strip().rstrip(".,;:")

        synonym_map = {
            "refrigerator": "fridge",
            "microwave oven": "microwave",
            "counter top": "countertop",
        }
        for old, new in synonym_map.items():
            s = s.replace(old, new)

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

        put_match = re.match(r"^(put|place)\s+.+\s+(in|into|on|onto)\s+(.+)$", s)
        if put_match:
            destination = put_match.group(3).strip().rstrip(".,;:")
            return f"put {destination}"

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

                non_receptacle_words = {
                    "egg", "mug", "cup", "bowl", "plate", "knife", "glass",
                    "potato", "apple", "bread", "toiletpaper", "laptop"
                }

                if target in non_receptacle_words and last_found_object is not None:
                    fixed.append(f"put {last_found_object}")
                else:
                    fixed.append(s)
                continue

            fixed.append(s)

        return fixed

    # ----------------------------
    # Scene facts / inventory
    # ----------------------------
    @staticmethod
    def extract_scene_facts_from_metadata(
        objects_metadata: List[Dict[str, Any]],
        task_description: Optional[str] = None,
        max_detailed_facts: int = 30
    ) -> Dict[str, Any]:
        """
        Return:
          {
            "all_object_names": [...],   # all object types in the scene
            "detailed_facts": [...]      # compact facts for the most useful objects
          }

        Design:
        - always give the model the name of ALL object types in the scene
        - only give detailed state/precondition facts for the most relevant objects
        """

        always_keep = {
            "Fridge", "Microwave", "CounterTop", "Cabinet", "Drawer",
            "StoveBurner", "StoveKnob", "Faucet", "Sink", "SinkBasin",
            "GarbageCan", "CoffeeMachine", "ToiletPaper", "Laptop",
            "Mug", "Cup", "Bowl", "Plate", "Knife", "Pan", "Pot",
            "Egg", "Bread"
        }

        task_words: Set[str] = set()
        if task_description:
            task_words = set(re.findall(r"[a-zA-Z]+", task_description.lower()))

        def is_task_relevant(obj_type: str) -> bool:
            if not obj_type:
                return False

            obj_lower = obj_type.lower()
            if obj_lower in task_words:
                return True

            aliases = {
                "cup": {"cup", "mug"},
                "mug": {"cup", "mug"},
                "sinkbasin": {"sink", "sinkbasin"},
                "sink": {"sink", "sinkbasin"},
                "garbagecan": {"garbagecan", "trashcan", "trash", "bin"},
                "toiletpaper": {"toiletpaper", "toilet", "paper"},
                "stoveknob": {"stoveknob", "stove", "knob"},
                "stoveburner": {"stoveburner", "stove", "burner"},
                "countertop": {"countertop", "counter"},
                "coffeemachine": {"coffeemachine", "coffee", "machine"},
            }

            if obj_lower in aliases:
                return len(aliases[obj_lower] & task_words) > 0

            return False

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for obj in objects_metadata:
            obj_type = obj.get("objectType")
            if not obj_type:
                continue
            grouped.setdefault(obj_type, []).append(obj)

        # ALL object names in the whole scene
        all_object_names = sorted(grouped.keys())

        # Detailed facts only for the most useful objects
        scored_types = []
        for obj_type, objs in grouped.items():
            score = 0
            if obj_type in always_keep:
                score += 2
            if is_task_relevant(obj_type):
                score += 5
            if any(o.get("visible", False) for o in objs):
                score += 2
            if any(o.get("parentReceptacles") for o in objs):
                score += 1
            if any(o.get("isOpen") is not None for o in objs):
                score += 1
            if any(o.get("isToggled") is not None for o in objs):
                score += 1

            scored_types.append((score, obj_type, objs))

        scored_types.sort(key=lambda x: (-x[0], x[1]))

        detailed_facts = []
        for _, obj_type, objs in scored_types[:max_detailed_facts]:
            obj = next((o for o in objs if o.get("visible", False)), objs[0])

            parent_receptacles = obj.get("parentReceptacles")
            is_open = obj.get("isOpen")
            is_toggled = obj.get("isToggled")
            is_picked = obj.get("isPickedUp")
            visible = obj.get("visible")

            details = ["present"]

            if visible is not None:
                details.append(f"visible={visible}")
            if parent_receptacles:
                details.append(f"inside/on={parent_receptacles}")
            if is_open is not None:
                details.append(f"open={is_open}")
            if is_toggled is not None:
                details.append(f"on={is_toggled}")
            if is_picked is not None:
                details.append(f"picked={is_picked}")

            detailed_facts.append(f"- {obj_type}: " + ", ".join(details))

        return {
            "all_object_names": all_object_names,
            "detailed_facts": detailed_facts,
        }

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

        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        # Save the exact image bytes sent to Gemini
        os.makedirs("model_input_images", exist_ok=True)
        save_path = os.path.join(
            "model_input_images",
            f"gemini_input_{len(os.listdir('model_input_images')):05d}.png"
        )
        with open(save_path, "wb") as f:
            f.write(img_bytes)

        image_part = Part.from_data(
            data=img_bytes,
            mime_type="image/png"
        )

        model = GenerativeModel(model_type)

        response = model.generate_content(
            [image_part, prompt]
        )

        return response.text if response.text else ""

    # ----------------------------
    # Table planners
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
        1) scene-understanding stage
        2) grounding stage with oracle objects + compact scene facts
        3) final planning stage

        Returns:
          enviroment_info, raw_plan_text, normalized_plan_steps
        """

        oracle_objects_text = self.format_oracle_objects(objs_from_scene)

        scene_info = self.extract_scene_facts_from_metadata(
            objects_metadata or [],
            task_description=self.task_description,
            max_detailed_facts=30
        )
        all_object_names_text = ", ".join(scene_info["all_object_names"])
        detailed_facts_text = "\n".join(scene_info["detailed_facts"]) if scene_info["detailed_facts"] else "- No detailed facts available."

        # ---- Stage 1: image relations / scene understanding ----
        prompt_env = f"""
You are an assistant that accurately describes the content of an image.

Task:
{self.task_description}

Your job:
1. Identify the main task-relevant objects visible in the image.
2. Write short relations between them.
3. Focus on relations such as:
   - inside
   - on
   - in front of
   - left of
   - right of
4. If a task-relevant object is not visible, say it is not visible in the current image only.

Return short structured statements only.
"""

        enviroment_agent_output = self._generate_from_image_and_text(prompt_env, model_type)

        # ---- Stage 2: grounding with oracle objects + scene inventory ----
        prompt_ground = f"""
You are an assistant that grounds a task in a scene using image evidence and oracle scene information.

Task:
{self.task_description}

Oracle object types provided by the benchmark:
{oracle_objects_text}

All object types present in the scene (source of truth):
{all_object_names_text}

Important compact scene facts:
{detailed_facts_text}

Rules:
- Use the all-object scene list as the source of truth for what exists in the scene.
- If an object is not visible in the image, it may still exist elsewhere in the scene.
- Do not say an object is absent if it appears in the scene object list.
- Use exact object names from the scene object list.
- Do not invent synonyms that are not in the scene object list.
- If Cup exists, use Cup. If Mug exists, use Mug.
- If an object is inside a closed receptacle, mention that access requires opening the receptacle first.
- Mention only task-relevant preconditions.

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

All object types present in the scene:
{all_object_names_text}

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
1. Use only objects that are in the scene object list.
2. Use exact object names from the scene object list.
3. Respect action preconditions.
4. If an object is inside a closed receptacle, first find and open that receptacle.
5. Only pick an object when it is accessible.
6. If an appliance must be off before opening, turn it off first.
7. If the task has multiple subtasks, complete all subtasks.
8. Return only the plan.
9. Return one action per line.
10. Keep actions atomic and executable.
11. For put actions, use only the destination receptacle/surface.
12. Valid examples:
    - put microwave
    - put countertop
    - put garbagecan
13. Do not output:
    - put egg
    - put mug
    - put cup in microwave
    - put egg on countertop

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
