"""
Finds the full LLM GGUF path from the Hugging Face cache.
"""

import os
import argparse

CACHE_DIR = "/runpod-volume/huggingface-cache/hub"


def find_model_path(model_name, gguf_in_repo="model.gguf"):
    """
    Find the path to a cached model.

    Args:
        model_name: The model name from Hugging Face

    Returns:
        The full path to the cached model, or None if not found
    """

    cache_name = model_name.replace("/", "--").lower()
    snapshots_dir = os.path.join(
        CACHE_DIR, f"models--{cache_name}", "snapshots"
    )

    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)

        if snapshots:
            return os.path.join(snapshots_dir, snapshots[0], gguf_in_repo)

    return None


def main():
    """
    Main function to find and print the model path.
    """

    parser = argparse.ArgumentParser(
        description="Find the full GGUF path from the Hugging Face cache."
    )
    parser.add_argument(
        "model", type=str, help="The model name from Hugging Face"
    )
    parser.add_argument(
        "path",
        type=str,
        help="The path to the GGUF file within the model repository",
    )
    args = parser.parse_args()

    model_path = find_model_path(args.model, args.path)
    print(model_path, end="")


if __name__ == "__main__":
    main()
