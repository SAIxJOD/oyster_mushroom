from pathlib import Path
from collections import Counter
import re


DATASET = Path("raw_dataset")

SPLITS = ["train", "valid", "test"]

CLASS_NAMES = {
    0: "Bacterial Blotch",
    1: "Dry Bubble",
    2: "Healthy",
    3: "Trichoderma",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def get_group_name(filename: str) -> str:
    name = Path(filename).stem

    # Remove Roboflow hash
    name = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", name)

    # Remove encoded extension
    name = re.sub(
        r"_(jpg|jpeg|png|bmp|webp)$",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Remove augmentation suffix
    name = re.sub(
        r"_aug_\d+$",
        "",
        name,
        flags=re.IGNORECASE
    )

    return name


def get_classes(label_file: Path) -> set[int]:
    classes = set()

    if not label_file.exists():
        return classes

    text = label_file.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return classes

    for line in text.splitlines():
        parts = line.split()

        if not parts:
            continue

        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue

        if class_id in CLASS_NAMES:
            classes.add(class_id)

    return classes


def main():
    groups = {}

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"

        if not image_dir.exists():
            continue

        for image in image_dir.iterdir():

            if not image.is_file():
                continue

            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            group = get_group_name(image.name)

            if group not in groups:
                groups[group] = {
                    "images": 0,
                    "splits": set(),
                    "classes": set()
                }

            groups[group]["images"] += 1
            groups[group]["splits"].add(split)

            label_file = label_dir / f"{image.stem}.txt"

            groups[group]["classes"].update(
                get_classes(label_file)
            )

    combinations = Counter()

    for info in groups.values():
        combo = tuple(sorted(info["classes"]))
        combinations[combo] += 1

    print("=" * 75)
    print("ORIGINAL GROUP CLASS COMBINATIONS")
    print("=" * 75)

    print(f"Total original groups: {len(groups)}")
    print()

    for combo, count in sorted(
        combinations.items(),
        key=lambda item: (-item[1], item[0])
    ):
        names = [
            CLASS_NAMES[c]
            for c in combo
        ]

        label = " + ".join(names)

        print(
            f"{label:<65} {count:>4} groups"
        )

    print()
    print("=" * 75)
    print("GROUP SIZE DISTRIBUTION")
    print("=" * 75)

    size_counter = Counter(
        info["images"]
        for info in groups.values()
    )

    for size, count in sorted(size_counter.items()):
        print(
            f"{size:>3} images per group : {count:>4} groups"
        )

    print()
    print("=" * 75)
    print("COMBINATION ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()