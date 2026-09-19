import os
import json
import numpy as np
from typing import List, Dict, Optional
from llm_sdk import Small_LLM_Model


class ConstrainedGenerator:
    def __init__(self, function_defs: Optional[Dict] = None) -> None:
        print("Initializing Small_LLM_Model (Qwen3-0.6B)...")
        self.model = Small_LLM_Model()

        print("Downloading/Locating vocab file via SDK...")
        self.vocab_path = self.model.get_path_to_vocab_file()

        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._load_vocabulary()

        self.function_defs = function_defs or {}
        self.function_names = (list(self.function_defs.keys())
                               if self.function_defs else [])

    def _load_vocabulary(self) -> None:
        if not os.path.exists(self.vocab_path):
            return
        try:
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        except Exception as e:
            print(f"Error loading vocabulary JSON: {e}")

    def _get_valid_next_characters(self, current_text: str) -> List[str]:
        # Entferne Whitespace für die Analyse
        clean = (current_text.replace(" ", "").replace("\n", "")
                 .replace("\r", ""))

        if not clean:
            return ["{"]

        target_prefix = '{"name":'
        if target_prefix.startswith(clean) and clean != target_prefix:
            return [target_prefix[len(clean)]]
        if clean == '{"name":':
            return ['"']

        if clean.startswith('{"name":"'):
            name_part = clean[len('{"name":"'):]
            if '"' not in name_part:
                allowed = set()
                for name in self.function_names:
                    if name.startswith(name_part):
                        if len(name) > len(name_part):
                            allowed.add(name[len(name_part)])
                        else:
                            allowed.add('"')
                return list(allowed) if allowed else ['"']
            else:
                closing_idx = name_part.index('"')
                remainder = name_part[closing_idx+1:]
                if not remainder:
                    return [","]
                args_prefix = ',"arguments":{'
                if (
                    args_prefix.startswith(remainder) and
                    remainder != args_prefix
                     ):
                    return [args_prefix[len(remainder)]]
                if remainder.startswith(args_prefix):
                    args_content = remainder[len(args_prefix):]
                    return (self._get_allowed_chars_for_arguments
                            (args_content, clean))
        return ['}']

    @staticmethod
    def _get_parameter_type(
        params: Dict, parameter_name: Optional[str]
         ) -> str:
        if parameter_name is None:
            raise RuntimeError("Kein aktiver Parameter vorhanden.")

        parameter = params.get(parameter_name)

        if parameter is None:
            raise RuntimeError(f"Unbekannter Parameter: {parameter_name!r}")

        if hasattr(parameter, "type"):
            return str(parameter.type)

        if isinstance(parameter, dict):
            return str(parameter.get("type"))

        raise RuntimeError(
            f"Parametertyp konnte nicht ermittelt werden: {parameter!r}"
        )

    def _get_allowed_chars_for_arguments(
            self, args_content: str, full_clean: str
            ) -> List[str]:
        name_start = full_clean.index('"name":"') + len('"name":"')
        name_end = full_clean.index('"', name_start)
        func_name = full_clean[name_start:name_end]

        params = self.function_defs.get(func_name, {}).get("parameters", {})
        param_names = list(params.keys())

        i = 0
        state = "EXPECT_KEY_START"

        current_key = ""
        used = set()

        while i < len(args_content):

            c = args_content[i]

            if state == "EXPECT_KEY_START":
                if c == '"':
                    current_key = ""
                    state = "READ_KEY"

            elif state == "READ_KEY":
                if c == '"':
                    active_key = current_key
                    used.add(current_key)
                    state = "EXPECT_COLON"
                else:
                    current_key += c

            elif state == "EXPECT_COLON":
                if c == ":":
                    state = "EXPECT_VALUE"

            elif state == "EXPECT_VALUE":
                parameter_type = self._get_parameter_type(params, active_key)

                if parameter_type == "string" and c == '"':
                    state = "READ_STRING"

                elif parameter_type == "number" and (c.isdigit() or c == "-"):
                    state = "READ_NUMBER"

                elif parameter_type == "boolean" and c in {"t", "f"}:
                    state = "READ_BOOLEAN"

            elif state == "READ_STRING":
                if c == '"':
                    state = "EXPECT_COMMA"

            elif state == "READ_NUMBER":
                if c == ",":
                    state = "EXPECT_KEY_START"
                    active_key = None
                elif c == "}":
                    state = "DONE"

            elif state == "READ_BOOLEAN":
                if c == ",":
                    state = "EXPECT_KEY_START"
                    active_key = None
                elif c == "}":
                    state = "DONE"

            elif state == "EXPECT_COMMA":
                if c == ",":
                    state = "EXPECT_KEY_START"

            i += 1

        remaining = [p for p in param_names if p not in used]

        if state == "EXPECT_KEY_START":

            if not remaining:
                return ["}"]

            return ['"']

        if state == "READ_KEY":

            allowed = set()

            for p in remaining:
                if p.startswith(current_key):
                    if len(current_key) == len(p):
                        allowed.add('"')
                    else:
                        allowed.add(p[len(current_key)])

            return list(allowed)

        if state == "EXPECT_COLON":
            return [":"]

        if state == "EXPECT_VALUE":
            parameter_type = self._get_parameter_type(params, active_key)

            if parameter_type == "number":
                return [
                    "0", "1", "2", "3", "4",
                    "5", "6", "7", "8", "9"
                ]

            if parameter_type == "string":
                return ['"']

            if parameter_type == "boolean":
                return ["t", "f"]

            raise RuntimeError(
                f"Nicht unterstützter Parametertyp: {parameter_type!r}"
            )

        if state == "READ_NUMBER":
            return [
                "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9",
                ".", ",", "}"
            ]

        if state == "READ_STRING":
            return list(
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                " _-. /\\'\":,"
            ) + ['"']

        if state == "READ_BOOLEAN":
            boolen_values = {
             "": ["t", "f"],
             "t": ["r"],
             "tr": ["u"],
             "tru": ["e"],
             "f": ["a"],
             "fa": ["l"],
             "fal": ["s"],
             "fals": ["e"],
             "true": [",", "}"],
             "false": [",", "}"],
            }
            value_start = args_content.rfind(":") + 1
            boolean_text = args_content[value_start:].strip()

            if "," in boolean_text:
                boolean_text = boolean_text.split(",")[-1].strip()
            return boolen_values.get(boolean_text, [])

        if state == "EXPECT_COMMA":

            if remaining:
                return [","]

            return ["}"]

        return ["}"]

    def generate_next_token(
            self, current_ids: List[int], prompt_length: int
            ) -> int:
        raw_logits = self.model.get_logits_from_input_ids(current_ids)
        logits = np.array(raw_logits, dtype=np.float32)

        generated_ids = current_ids[prompt_length:]
        current_text = self.model.decode(generated_ids)

        allowed_chars = self._get_valid_next_characters(current_text)

        masked_logits = np.full_like(logits, -1e9)

        for token_id, token_str in self.inverse_vocab.items():
            cleaned_token = token_str.replace("Ġ", "").replace("Ċ", "")

            if not cleaned_token:
                continue

            if cleaned_token in allowed_chars:
                masked_logits[token_id] = logits[token_id]

        best = int(np.argmax(masked_logits))

        print("Allowed:", allowed_chars)
        print("Chosen token:",
              repr(self.inverse_vocab[best])
              )
        return int(np.argmax(masked_logits))
