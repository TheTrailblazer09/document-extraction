"""
Verify that the environment and datasets are set up correctly.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    status = "OK" if ok else "WARN"
    print(f"  [{status}] Python {v.major}.{v.minor}.{v.micro} (need 3.10+)")
    return ok


def check_package(name, import_name=None):
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "installed")
        print(f"  [OK]   {name} ({version})")
        return True
    except ImportError:
        print(f"  [FAIL] {name} not installed")
        return False


def check_dataset(name, path):
    exists = path.exists() and any(path.iterdir())
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {name} -> {path}")
    return exists


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [OK]   CUDA available: {gpu_name}")
            return True
        else:
            print(f"  [WARN] No CUDA GPU detected (training will be slow)")
            return False
    except ImportError:
        print(f"  [FAIL] PyTorch not installed")
        return False


def main():
    print("\n" + "=" * 60)
    print("Environment Check")
    print("=" * 60)

    print("\nPython:")
    check_python()

    print("\nCore packages:")
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("wandb", "wandb"),
        ("pytesseract", "pytesseract"),
        ("albumentations", "albumentations"),
        ("seqeval", "seqeval"),
        ("PIL", "PIL"),
        ("cv2", "cv2"),
        ("yaml", "yaml"),
    ]
    all_packages_ok = all(check_package(n, i) for n, i in packages)

    print("\nGPU:")
    check_gpu()

    print("\nDatasets:")
    datasets_ok = True
    for name, subdir in [("CORD", "cord/hf_dataset"), ("FUNSD", "funsd/hf_dataset"), ("SROIE", "sroie")]:
        if not check_dataset(name, DATA_DIR / subdir):
            datasets_ok = False

    print("\n" + "=" * 60)
    if all_packages_ok and datasets_ok:
        print("All checks passed! Ready to go.")
    else:
        if not all_packages_ok:
            print("Some packages missing. Run: pip install -r requirements.txt")
        if not datasets_ok:
            print("Some datasets missing. Run: python scripts/download_data.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
