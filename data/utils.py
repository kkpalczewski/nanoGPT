"""Common utilities for dataset preparation scripts."""
import os
import argparse


def get_output_dir(dataset_name: str) -> str:
    """Get output directory from LAMBDA_SHARED_STORAGE env var or fallback to local."""
    shared_storage = os.environ.get('LAMBDA_SHARED_STORAGE')
    if shared_storage:
        return os.path.join(shared_storage, dataset_name)
    return os.path.join(os.path.dirname(__file__), dataset_name)


def files_exist(output_dir: str, files: list[str] = None) -> bool:
    """Check if required files already exist in output directory."""
    if files is None:
        files = ['train.bin', 'val.bin']
    return all(os.path.exists(os.path.join(output_dir, f)) for f in files)


def remove_files(output_dir: str, files: list[str]) -> None:
    """Remove specified files from output directory if they exist."""
    for f in files:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")


def parse_args(description: str) -> argparse.Namespace:
    """Parse common command line arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force re-download and re-process even if files exist')
    return parser.parse_args()


def setup_output_dir(dataset_name: str, force: bool, extra_files: list[str] = None) -> str | None:
    """
    Setup output directory and check if processing is needed.
    
    Returns output_dir if processing should proceed, None if files exist and force=False.
    """
    output_dir = get_output_dir(dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    files_to_check = ['train.bin', 'val.bin']
    if extra_files:
        files_to_check.extend(extra_files)

    if files_exist(output_dir) and not force:
        print("train.bin and val.bin already exist. Use --force to re-process.")
        return None

    if force and files_exist(output_dir):
        print("Force flag set. Removing existing files...")
        remove_files(output_dir, files_to_check)

    return output_dir
