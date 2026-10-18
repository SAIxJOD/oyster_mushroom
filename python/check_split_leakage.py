from pathlib import Path
from collections import defaultdict
import re


DATASET = Path("raw_dataset")
SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def normalized_group(filename: str) -> str:
    """
    Convert related filenames to a common source-image group.

    Examples:
        img_144_aug_1_jpg.rf.hash.jpg -> img_144
        img_144_aug_2_jpg.rf.hash.jpg -> img_144
        img_144_jpg.rf.hash.jpg        -> img_144
    """
    name = Path(filename).stem

    # Remove Roboflow hash suffix such as .rf.abcdef...
    name = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", name)

    # Remove image-format suffix encoded in filename.
    name = re.sub(r"_jpg$", "", name)
    name = re.sub(r"_jpeg$", "", name)
    name = re.sub(r"_png$", "", name)

    # Remove augmentation suffix.
    name = re.sub(r"_aug_\d+$", "", name)

    return name


def main():
    groups = defaultdict(set)
    counts = defaultdict(int)

    for split in SPLITS:
        image_dir = DATASET / split / "images"

        if not image_dir.exists():
            print(f"Missing folder: {image_dir}")
            continue

        for image in image_dir.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                key = normalized_group(image.name)
                groups[key].add(split)
                counts[(split, key)] += 1

    cross_split = {
        key: splits
        for key, splits in groups.items()
        if len(splits) > 1
    }

    print("=" * 70)
    print("DATASET SPLIT LEAKAGE CHECK")
    print("=" * 70)

    for split in SPLITS:
        total = sum(1 for key, value in counts.items() if key[0] == split)
        print(f"{split.upper():6} images: {total}")

    print("\nSource groups appearing in multiple splits:")
    print("-" * 70)

    if not cross_split:
        print("NONE FOUND")
    else:
        print(f"Groups crossing splits: {len(cross_split)}")

        for key, splits in list(sorted(cross_split.items()))[:50]:
            print(f"{key} -> {', '.join(sorted(splits))}")

    print("\n" + "=" * 70)
    print("CHECK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()