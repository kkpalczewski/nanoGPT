"""Common utilities for dataset preparation scripts."""
import os
import subprocess
import argparse
from dotenv import load_dotenv


def get_local_dir(dataset_name: str) -> str:
    """Get local data directory."""
    return os.path.join('data', dataset_name)


def get_nfs_dir(dataset_name: str) -> str:
    """Get NFS directory from LAMBDA_SHARED_STORAGE."""
    load_dotenv()
    shared_storage = os.environ.get('LAMBDA_SHARED_STORAGE')
    if shared_storage:
        return os.path.join(shared_storage, dataset_name)
    return None


def files_exist(output_dir: str, files: list[str]) -> bool:
    """Check if required files exist in output directory."""
    if not output_dir or not os.path.exists(output_dir):
        return False
    return all(os.path.exists(os.path.join(output_dir, f)) for f in files)


def copy_from_nfs(nfs_dir: str, local_dir: str, files: list[str]) -> None:
    """Copy files from NFS to local using rsync."""
    os.makedirs(local_dir, exist_ok=True)
    for f in files:
        src = os.path.join(nfs_dir, f)
        if os.path.exists(src):
            dst = os.path.join(local_dir, f)
            subprocess.run(['rsync', '-az', '--progress', src, dst], check=True)


def copy_to_nfs(local_dir: str, nfs_dir: str, files: list[str]) -> None:
    """Copy files from local to NFS using rsync."""
    os.makedirs(nfs_dir, exist_ok=True)
    for f in files:
        src = os.path.join(local_dir, f)
        if os.path.exists(src):
            dst = os.path.join(nfs_dir, f)
            subprocess.run(['rsync', '-az', '--progress', src, dst], check=True)


def parse_args(description: str) -> argparse.Namespace:
    """Parse common command line arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--force', action='store_true', default=False,
                        help='Force re-download and re-process even if files exist')
    return parser.parse_args()


def check_and_setup(dataset_name: str, force: bool, required_files: list[str]) -> tuple[str, str] | None:
    """
    Check if files exist (local first, then NFS), setup directories.
    
    Returns (local_dir, nfs_dir) if processing needed, None if files already exist.
    """
    local_dir = get_local_dir(dataset_name)
    nfs_dir = get_nfs_dir(dataset_name)
    
    # Check if files already exist locally
    if files_exist(local_dir, required_files) and not force:
        print(f"Files already exist in {local_dir}")
        return None
    
    # Check NFS - if files exist there, copy to local
    if nfs_dir and files_exist(nfs_dir, required_files) and not force:
        print(f"Files found on NFS, copying to local...")
        copy_from_nfs(nfs_dir, local_dir, required_files)
        return None
    
    # Files don't exist - need to process
    if force:
        print("Force flag set, will re-process")
        for f in required_files:
            path = os.path.join(local_dir, f)
            if os.path.exists(path):
                os.remove(path)
            if nfs_dir:
                path = os.path.join(nfs_dir, f)
                if os.path.exists(path):
                    os.remove(path)
    os.makedirs(local_dir, exist_ok=True)
    return local_dir, nfs_dir
