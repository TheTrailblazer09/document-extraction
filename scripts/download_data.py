"""
Download all datasets for the project.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dataset cord      # download only CORD
    python scripts/download_data.py --dataset funsd     # download only FUNSD
    python scripts/download_data.py --dataset sroie     # download only SROIE
"""

import argparse
import os
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def download_cord():
    """Download CORD v2 from Hugging Face (~1.5GB)."""
    print("=" * 60)
    print("Downloading CORD v2 dataset...")
    print("Source: https://huggingface.co/datasets/naver-clova-ix/cord-v2")
    print("=" * 60)

    from datasets import load_dataset

    cord_dir = DATA_DIR / "cord"
    cord_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("naver-clova-ix/cord-v2")

    # Save to disk for offline use
    dataset.save_to_disk(str(cord_dir / "hf_dataset"))

    print(f"\nCORD downloaded successfully!")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Val:   {len(dataset['validation'])} examples")
    print(f"  Test:  {len(dataset['test'])} examples")
    print(f"  Saved to: {cord_dir / 'hf_dataset'}")
    print()


def download_funsd():
    """Download FUNSD dataset (~150MB)."""
    print("=" * 60)
    print("Downloading FUNSD dataset...")
    print("Source: https://guillaumejaume.github.io/FUNSD/")
    print("=" * 60)

    from datasets import load_dataset

    funsd_dir = DATA_DIR / "funsd"
    funsd_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("nielsr/funsd-layoutlmv3")

    dataset.save_to_disk(str(funsd_dir / "hf_dataset"))

    print(f"\nFUNSD downloaded successfully!")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Test:  {len(dataset['test'])} examples")
    print(f"  Saved to: {funsd_dir / 'hf_dataset'}")
    print()


def download_sroie():
    """
    Download SROIE dataset from Kaggle (~200MB).

    Requires Kaggle API credentials:
      1. Go to kaggle.com -> Account -> Create New API Token
      2. Save kaggle.json to ~/.kaggle/kaggle.json
      3. chmod 600 ~/.kaggle/kaggle.json
    """
    print("=" * 60)
    print("Downloading SROIE dataset from Kaggle...")
    print("Source: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2")
    print("=" * 60)

    sroie_dir = DATA_DIR / "sroie"
    sroie_dir.mkdir(parents=True, exist_ok=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n*** Kaggle API credentials not found! ***")
        print("To set up:")
        print("  1. Go to kaggle.com -> Account -> Create New API Token")
        print("  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("  3. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("  4. Re-run this script")
        print()
        print("Alternatively, download manually from:")
        print("  https://www.kaggle.com/datasets/urbikn/sroie-datasetv2")
        print(f"  Extract to: {sroie_dir}")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "urbikn/sroie-datasetv2",
            path=str(sroie_dir),
            unzip=True,
        )
        print(f"\nSROIE downloaded successfully!")
        print(f"  Saved to: {sroie_dir}")
    except Exception as e:
        print(f"\nError downloading SROIE: {e}")
        print("You can download manually from:")
        print("  https://www.kaggle.com/datasets/urbikn/sroie-datasetv2")
        print(f"  Extract to: {sroie_dir}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Download project datasets")
    parser.add_argument(
        "--dataset",
        choices=["cord", "funsd", "sroie", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaders = {
        "cord": download_cord,
        "funsd": download_funsd,
        "sroie": download_sroie,
    }

    if args.dataset == "all":
        for name, fn in downloaders.items():
            try:
                fn()
            except Exception as e:
                print(f"ERROR downloading {name}: {e}")
                print("Continuing with remaining datasets...\n")
    else:
        downloaders[args.dataset]()

    print("=" * 60)
    print("Done! Run 'python scripts/verify_setup.py' to check everything.")
    print("=" * 60)


if __name__ == "__main__":
    main()
