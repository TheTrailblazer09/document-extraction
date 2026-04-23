from data_loading import load_dataset

import matplotlib.pyplot as plt


def verify_dataset(name):
    dataset = load_dataset(name)
    sample = dataset.train[0]
    print(f"  keys:       {list(sample.keys())}")
    print(f"  image size: {sample['image'].size}")
    print()
    return dataset


def main():
    # --- CORD ---
    print("=== CORD ===")
    cord = verify_dataset("cord")
    print(f"  ground_truth preview: {cord.train[0]['ground_truth'][:80]}...\n")

    # --- FUNSD ---
    print("=== FUNSD ===")
    funsd = verify_dataset("funsd")
    sample = funsd.train[0]
    print(f"  tokens preview:   {sample['tokens'][:5]}")
    print(f"  ner_tags preview: {sample['ner_tags'][:5]}\n")

    # --- SROIE ---
    print("=== SROIE ===")
    sroie = verify_dataset("sroie")
    sample = sroie.train[0]
    print(f"  entities:     {sample['entities']}")
    print(f"  boxes (first): {sample['boxes'][0]}\n")

    # --- Display an image from each dataset as a sanity check ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (dataset, title) in zip(axes, [(cord, "CORD"), (funsd, "FUNSD"), (sroie, "SROIE")]):
        ax.imshow(dataset.train[0]["image"])
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()