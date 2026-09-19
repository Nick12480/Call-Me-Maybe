"""Utilities for constrained generation with the provided LLM SDK."""

from typing import Any, Dict, List
import json
import re
import numpy as np
from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition


def encode_prompt(
    model: Small_LLM_Model,
    prompt: str,
) -> List[int]:
    """Encode a prompt into a flat list of token IDs.

    Args:
        model: Initialized language model.
        prompt: Text to encode.

    Returns:
        A flat list containing the prompt token IDs.
    """
    encoded = model.encode(prompt)
    token_ids: List[int] = encoded.squeeze(0).tolist()
    return token_ids


def get_best_next_token(
    model: Small_LLM_Model,
    input_ids: List[int],
) -> int:
    """Return the token ID with the highest model logit.

    Args:
        model: Initialized language model.
        input_ids: Current sequence of token IDs.

    Returns:
        The token ID with the highest logit.

    Raises:
        RuntimeError: If the model returns invalid logits.
    """
    raw_logits = model.get_logits_from_input_ids(input_ids)
    logits = np.asarray(raw_logits, dtype=np.float32)

    if logits.ndim != 1:
        raise RuntimeError(
            f"Expected one-dimensional logits, got shape {logits.shape}."
        )

    return int(np.argmax(logits))


def build_model_prompt(
    user_prompt: str,
    functions: List[FunctionDefinition],
) -> str:
    """Build a model prompt containing all available functions.

    Args:
        user_prompt: Natural-language request to process.
        functions: Available function definitions.

    Returns:
        A complete instruction prompt for the model.
    """
    function_lines: List[str] = []

    for function in functions:
        parameter_lines = [
            f"    - {name}: {definition.type}"
            for name, definition in function.parameters.items()
        ]

        parameters = "\n".join(parameter_lines)

        function_lines.append(
            f"- {function.name}\n"
            f"  Description: {function.description}\n"
            f"  Parameters:\n{parameters}"
        )

    function_context = "\n\n".join(function_lines)

    return (
        "Choose exactly one function for the request.\n"
        "Extract all required arguments.\n"
        "Copy string arguments EXACTLY from the request.\n"
        "Do not summarize, rewrite, normalize, shorten, "
        "or modify string arguments.\n"
        "For string parameters, preserve every character exactly, including "
        "spaces, punctuation, quotes, braces, slashes,"
        " apostrophes, and capitalization.\n"
        "For regex arguments, output the shortest suitable regex.\n"
        'Output only JSON: {"name":"...","parameters":{...}}\n\n'
        f"{function_context}\n\n"
        f"Request: {user_prompt}\n"
        "JSON:"
        )


def get_allowed_string_characters(
    value_text: str,
    terminator: str,
) -> List[str]:
    """Return valid continuations for a non-empty JSON string.

    Args:
        value_text: String value generated so far.
        terminator: Character following the completed string.

    Returns:
        Characters that may validly continue the string.
    """
    letters_and_digits = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
    )

    allowed_content = list(
        letters_and_digits
        + " _-.'"
        + "[]+*?^$()|/,=:;<>!%&\\"
    )

    if value_text == "":
        return ['"']

    if not value_text.startswith('"'):
        return []

    string_content = value_text[1:]

    if '"' not in string_content:
        if string_content == "":
            return list(
                letters_and_digits
                + "_-[/"
                + "(^*"
            )

        allowed = allowed_content.copy()

        valid_final_characters = (
            letters_and_digits
            + "_-].+*?)$"
            + "/=,:;<>!%&\\"
        )

        if string_content[-1] in valid_final_characters:
            allowed.append('"')

        return allowed

    closing_quote_index = string_content.index('"')
    remainder = string_content[
        closing_quote_index + 1:
    ]

    if remainder == "":
        return [terminator]

    if terminator == "}" and remainder == "}":
        return ["}"]

    return []


NUMBER_PATTERN = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
)


def is_complete_number(value_text: str) -> bool:
    """Return whether text is a complete supported JSON number."""
    return NUMBER_PATTERN.fullmatch(value_text) is not None


def get_allowed_number_characters(
    value_text: str,
    terminator: str,
) -> List[str]:
    """Return valid continuations for a supported JSON number."""
    digits = list("0123456789")

    if value_text == "":
        return digits + ["-"]

    if value_text == "-":
        return digits

    unsigned_text = value_text.removeprefix("-")

    if "." not in unsigned_text:
        if not unsigned_text.isdigit():
            return []

        # JSON forbids leading zeros such as 01.
        if len(unsigned_text) > 1 and unsigned_text.startswith("0"):
            return []

        if unsigned_text == "0":
            return [".", terminator]

        return digits + [".", terminator]

    integer_part, decimal_part = unsigned_text.split(".", 1)

    if not integer_part.isdigit():
        return []

    if len(integer_part) > 1 and integer_part.startswith("0"):
        return []

    if decimal_part == "":
        return digits

    if decimal_part.isdigit():
        return digits + [terminator]

    return []


def get_allowed_integer_characters(
        value_text: str, terminator: str
) -> List[str]:

    digits = list("0123456789")

    if value_text == "":
        return digits + ["-"]

    if value_text == "-":
        return digits

    if not value_text.isdigit():
        return []

    if len(value_text) > 1 and value_text.startswith("0"):
        return []

    return digits + [terminator]


def get_allowed_parameter_characters(
    parameter_text: str,
    function: FunctionDefinition,
) -> List[str]:
    """Return valid continuations for the parameters object.

    Args:
        parameter_text: Generated text inside the parameters object.
        function: Function selected by the language model.

    Returns:
        Characters that may validly continue the JSON document.
    """
    parameter_names = get_parameter_names(function)

    if not parameter_names:
        if parameter_text == "":
            return ["}"]

        if parameter_text == "}":
            return ["}"]

        return []

    cursor = 0

    for index, parameter_name in enumerate(parameter_names):
        parameter_prefix = f'"{parameter_name}":'
        remaining_text = parameter_text[cursor:]

        # Parameter name is currently being generated.
        if len(remaining_text) < len(parameter_prefix):
            if not parameter_prefix.startswith(remaining_text):
                return []

            next_index = len(remaining_text)
            return [parameter_prefix[next_index]]

        if not remaining_text.startswith(parameter_prefix):
            return []

        cursor += len(parameter_prefix)
        remaining_text = parameter_text[cursor:]

        is_last_parameter = index == len(parameter_names) - 1
        terminator = "}" if is_last_parameter else ","

        parameter_type = (
            function.parameters[parameter_name].type
        )

        if parameter_type == "string":
            if remaining_text == "":
                return ['"']

            if not remaining_text.startswith('"'):
                return []

            closing_quote_index = remaining_text.find(
                '"',
                1,
            )

            if closing_quote_index == -1:
                return get_allowed_string_characters(
                    remaining_text,
                    terminator,
                )

            string_content = remaining_text[
                1:closing_quote_index
            ]

            if string_content == "":
                return []

            text_after_string = remaining_text[
                closing_quote_index + 1:
            ]

            if text_after_string == "":
                return [terminator]

            if not text_after_string.startswith(terminator):
                return []

            cursor += (
                closing_quote_index
                + 1
                + len(terminator)
            )
            continue

        if parameter_type == "number":
            terminator_index = remaining_text.find(
                terminator
            )

            if terminator_index == -1:
                return get_allowed_number_characters(
                    remaining_text,
                    terminator,
                )

            number_text = remaining_text[
                :terminator_index
            ]

            if not is_complete_number(number_text):
                return []

            cursor += terminator_index + len(terminator)
            continue

        if parameter_type == "integer":
            terminator_index = remaining_text.find(
                terminator
            )

            if terminator_index == -1:
                return get_allowed_integer_characters(
                    remaining_text,
                    terminator,
                )

            integer_text = remaining_text[
                :terminator_index
            ]

            if not is_complete_number(integer_text):
                return []

            cursor += terminator_index + len(terminator)
            continue

        if parameter_type == "boolean":
            if remaining_text == "":
                return ["t", "f"]

            boolean_values = {
                "t": "true",
                "tr": "true",
                "tru": "true",
                "f": "false",
                "fa": "false",
                "fal": "false",
                "fals": "false",
            }

            if remaining_text in boolean_values:
                target = boolean_values[remaining_text]

                if len(remaining_text) < len(target):
                    return [target[len(remaining_text)]]

            if remaining_text in {"true", "false"}:
                text_after_bollean = remaining_text[
                    len(remaining_text):
                ]

                if text_after_bollean == "":
                    return [terminator]

                if not text_after_bollean.startswith(terminator):
                    return []

                cursor += (
                    len(remaining_text) + len(terminator)
                )
                continue

            return []

        return []

    trailing_text = parameter_text[cursor:]

    if trailing_text == "":
        return ["}"]

    return []


def get_allowed_next_characters(
    generated_text: str,
    functions: List[FunctionDefinition],
) -> List[str]:
    """Return characters that may validly continue the JSON output.

    This temporary implementation supports functions containing one or two
    positive integer parameters.

    Args:
        generated_text: Text generated after the model prompt.
        functions: Available function definitions.

    Returns:
        Characters that may validly continue the generated text.
    """
    name_prefix = '{"name":"'
    parameters_prefix = ',"parameters":{'
    compact_text = generated_text.lstrip()

    function_names = [
        function.name
        for function in functions
    ]

    # Generate the fixed beginning: {"name":"
    if compact_text == name_prefix:
        return _get_function_name_characters(
            "",
            function_names,
        )

    if name_prefix.startswith(compact_text):
        next_index = len(compact_text)
        return [name_prefix[next_index]]

    if not compact_text.startswith(name_prefix):
        return []

    name_part = compact_text[len(name_prefix):]

    # Generate a valid function name.
    if '"' not in name_part:
        return _get_function_name_characters(
            name_part,
            function_names,
        )

    closing_quote_index = name_part.index('"')
    selected_name = name_part[:closing_quote_index]
    remainder = name_part[closing_quote_index + 1:]

    if selected_name not in function_names:
        return []

    selected_function = next(
        function
        for function in functions
        if function.name == selected_name
    )

    if remainder.endswith("}}"):
        return []

    if parameters_prefix.startswith(remainder):
        if remainder == parameters_prefix:
            parameter_names = list(
                selected_function.parameters.keys()
            )

            if not parameter_names:
                return ["}"]

            first_parameter_prefix = f'"{parameter_names[0]}":'
            return [first_parameter_prefix[0]]

        next_index = len(remainder)
        return [parameters_prefix[next_index]]

    if not remainder.startswith(parameters_prefix):
        return []

    parameter_text = remainder[len(parameters_prefix):]

    return get_allowed_parameter_characters(
        parameter_text,
        selected_function
    )


def _get_function_name_characters(
    current_name: str,
    function_names: List[str],
) -> List[str]:
    """Return valid next characters for a function-name prefix.

    Args:
        current_name: Function-name prefix generated so far.
        function_names: Names of all available functions.

    Returns:
        Valid next characters for the current prefix.
    """
    allowed: set[str] = set()

    for function_name in function_names:
        if not function_name.startswith(current_name):
            continue

        if len(current_name) == len(function_name):
            allowed.add('"')
        else:
            allowed.add(function_name[len(current_name)])

    return sorted(allowed)


def get_token_text(
    model: Small_LLM_Model,
    token_id: int,
) -> str:
    """Decode one token ID into text.

    Args:
        model: Initialized language model.
        token_id: Token identifier to decode.

    Returns:
        Decoded token text.
    """
    return model.decode([token_id])


def get_parameter_names(
    function: FunctionDefinition,
) -> List[str]:
    """Return parameter names in their definition order.

    Args:
        function: Selected function definition.

    Returns:
        Parameter names in deterministic generation order.
    """
    return list(function.parameters.keys())


def extract_string_candidates(
        user_promt: str, parameter_name: str
        ) -> list:

    if parameter_name == "path":
        matches = re.findall(
            r'(?<!\S)(?:/[^\s]+|[A-Za-z]:\\[^\s]+)',
            user_promt
        )
        return matches

    if parameter_name in {"query", "template"}:

        candidates: List[str] = []

        candidates.extend(
            re.findall(r"'([^']*)'", user_promt)
        )

        candidates.extend(
            re.findall(r"'([^']*)'", user_promt)
        )

        return candidates

    return []


def token_is_valid(
    token_text: str,
    generated_text: str,
    functions: List[FunctionDefinition],
) -> bool:

    if token_text == "":
        return False

    trial_text = generated_text

    for character in token_text:
        allowed_characters = get_allowed_next_characters(
            trial_text,
            functions
        )
        if character not in allowed_characters:
            return False

        trial_text += character

    return True


def select_best_allowed_token(
    model: Small_LLM_Model,
    input_ids: List[int],
    generated_text: str,
    functions: List[FunctionDefinition],
) -> int:
    """Select the highest-scoring valid token.

    Args:
        model: Initialized language model.
        input_ids: Complete token sequence generated so far.
        generated_text: Decoded output generated so far.
        functions: Available function definitions.

    Returns:
        Identifier of the highest-scoring valid token.

    Raises:
        RuntimeError: If logits are invalid or no valid token exists.
    """
    raw_logits = model.get_logits_from_input_ids(input_ids)
    logits = np.asarray(raw_logits, dtype=np.float32)

    if logits.ndim != 1:
        raise RuntimeError(
            f"Expected one-dimensional logits, got shape {logits.shape}."
        )

    token_ids_by_score = np.argsort(logits)[::-1]

    for raw_token_id in token_ids_by_score:
        token_id = int(raw_token_id)
        token_text = model.decode([token_id])

        if token_is_valid(
            token_text,
            generated_text,
            functions,
        ):
            return token_id

    raise RuntimeError(
        "No valid token is available for the current decoder state."
    )


def validate_generated_call(
        generated_text: str,
        functions: List[FunctionDefinition]
) -> Dict[str, Any]:

    try:
        parsed = json.loads(generated_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Generated output is not valid JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "Generated output must be a JSON object."
        )

    if set(parsed.keys()) != {"name", "parameters"}:
        raise ValueError(
            "Generated output must contain exactly "
            "'name' and 'parameters'."
        )

    function_name = parsed["name"]
    parameters = parsed["parameters"]

    if not isinstance(function_name, str):
        raise ValueError(
            "The 'name' field must be a string."
        )

    if not isinstance(parameters, dict):
        raise ValueError(
            "The 'parameters' field must be an object."
        )

    definitions_by_name = {
        function.name: function
        for function in functions
    }

    if function_name not in definitions_by_name:
        raise ValueError(
            f"Unknown function name: {function_name!r}."
        )

    selected_function = definitions_by_name[function_name]
    expected_parameters = selected_function.parameters

    if set(parameters.keys()) != set(expected_parameters.keys()):
        raise ValueError(
            "Generated parameter names do not match "
            f"the definition of {function_name!r}."
        )

    for parameter_name, definition in expected_parameters.items():
        value = parameters[parameter_name]

        if definition.type == "number":
            if (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                 ):
                raise ValueError(
                    f"Parameter {parameter_name!r} "
                    "must be a number."
                )

            parameters[parameter_name] = float(value)

        elif definition.type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"Parameter {parameter_name!r} "
                    "must be an integer."
                )

        elif definition.type == "string":
            if not isinstance(value, str):
                raise ValueError(
                    f"Parameter {parameter_name!r} "
                    "must be a string."
                )

        elif definition.type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Parameter {parameter_name!r} "
                    "must be a boolean."
                )

        else:
            raise ValueError(
                f"Unsupported parameter type: "
                f"{definition.type!r}."
            )

    return parsed


def generate_function_call(
        model: Small_LLM_Model,
        user_prompt: str,
        functions: List[FunctionDefinition],
        max_steps: int = 100,
) -> Dict[str, Any]:

    model_prompt = build_model_prompt(
        user_prompt,
        functions
    )

    input_ids = encode_prompt(
        model,
        model_prompt,
    )

    generated_text = generate_constrained_prefix(
        model,
        input_ids,
        functions,
        max_steps=max_steps
    )

    return validate_generated_call(
        generated_text,
        functions
    )


def generate_constrained_prefix(
    model: Small_LLM_Model,
    input_ids: List[int],
    functions: List[FunctionDefinition],
    max_steps: int = 100,
) -> str:
    """Generate a constrained function-call JSON document."""
    generated_ids: List[int] = []
    generated_text = ""

    for _ in range(max_steps):
        allowed = get_allowed_next_characters(
            generated_text,
            functions,
        )

        if not allowed:
            try:
                json.loads(generated_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Decoder reached an invalid dead end at:\n"
                    f"{generated_text!r}"
                ) from exc

            return generated_text

        selected_token_id = select_best_allowed_token(
            model,
            input_ids + generated_ids,
            generated_text,
            functions,
        )

        selected_text = model.decode(
            [selected_token_id]
        )

        generated_ids.append(selected_token_id)
        generated_text += selected_text

    raise RuntimeError(
        f"Constrained generation exceeded {max_steps} steps."
    )
