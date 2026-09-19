import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from functools import wraps

from llm_sdk import Small_LLM_Model

from src.decoder import generate_function_call
from src.parser import (
    load_functions_definition,
    load_test_prompts,
)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Casting {func.__name__}...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"time: {end - start:.3f} seconds")
        return result
    return wrapper


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate function calls for all test prompts."
        )
    )

    parser.add_argument(
        "--functions-definition",
        default="moulinette/data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        default="moulinette/data/input/function_calling_tests.json",
        help="Path to the test prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of generated tokens per prompt.",
    )

    return parser.parse_args()


@timer
def main() -> None:
    """Process all prompts and write validated function calls."""
    args = parse_arguments()

    functions = load_functions_definition(
        args.functions_definition
    )

    if functions is None:
        print(
            "Could not load function definitions.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts = load_test_prompts(args.input)

    if prompts is None:
        print(
            "Could not load test prompts.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = Small_LLM_Model()
    results: List[Dict[str, Any]] = []

    for index, test_prompt in enumerate(prompts):
        print(
            f"[{index + 1}/{len(prompts)}] "
            f"Processing: {test_prompt.prompt}"
        )

        try:
            function_call = generate_function_call(
                model,
                test_prompt.prompt,
                functions,
                max_steps=args.max_steps,
            )
        except (RuntimeError, ValueError) as exc:
            print(
                f"Failed to process prompt "
                f"{test_prompt.prompt!r}: {exc}",
                file=sys.stderr,
            )
            continue

        print(
            "D.B.GENERATED:",
            json.dumps(function_call, ensure_ascii=False)
        )

        results.append(
            {
                "prompt": test_prompt.prompt,
                "name": function_call["name"],
                "parameters": function_call["parameters"],
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                results,
                output_file,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as exc:
        print(
            f"Could not write output file: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Wrote {len(results)} results to "
        f"{output_path}."
    )


if __name__ == "__main__":
    main()
