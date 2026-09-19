import json
import os
from typing import List, Optional
from pydantic import ValidationError
from src.models import TestPrompt, FunctionDefinition


def load_test_prompts(file_path: str) -> Optional[List[TestPrompt]]:
    """Loads and validates the natural language test prompts from a JSON file.

    Args:
        file_path: The filesystem path to the function_calling_tests.json file.

    Returns:
        A list of validated TestPrompt models, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: Promt input file not found at {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("Error: Prompt input file must contain a JSON array.")
            return None

        return [TestPrompt(**item) for item in data]

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON syntax in prompt file: {e}")
        return None
    except ValidationError as e:
        print(f"Error: Schema validation failed for prompt file: {e}")
        return None
    except Exception as e:
        print("Error: An unexpected error "
              f"occurred while reading prompts: {e}")
        return None


def load_functions_definition(
        file_path: str) -> Optional[List[FunctionDefinition]]:
    """Loads and validates the available function definitions from a JSON file.

    Args:
        file_path: The filesystem path to the functions_definition.json file.

    Returns:
        A list of validated FunctionDefinition models, or None if an error occurs.  # noqa: E501
    """
    if not os.path.exists(file_path):
        print(f"Error: Functions definition file not found at {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8")as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("Error: Functions definition "
                  "file must contain a JSON array.")
            return None

        return [FunctionDefinition(**item) for item in data]

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON syntax in functions definition file: {e}")
        return None
    except ValidationError as e:
        print(f"Error: Schema validation failed for functions definition: {e}")
        return None
    except Exception as e:
        print("Error: An unexpected error "
              f"occurred while reading definitions: {e}")
        return None
