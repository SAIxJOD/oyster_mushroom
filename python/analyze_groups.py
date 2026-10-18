from pathlib import Path
from collections import defaultdict, Counter
import re


DATASET = Path("raw_dataset")

CLASS_NAMES = {
    0: "Bacterial Blotch",
    1: "Dry Bubble",
    2: "Healthy",
    3: "Trichoderma",
}

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def get_group_name(filename: str) -> str:
    """
    Convert augmented Roboflow filenames into their original image group.

    Example:
        img_144_aug_1_jpg.rf.xxxxx.jpg
        img_144_aug_2_jpg.rf.xxxxx.jpg

    Both become:
        img_144
    """

    name = Path(filename).stem

    # Remove Roboflow hash
    name = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", name)

    # Remove encoded image extension
    name = re.sub(r"_(jpg|jpeg|png|bmp|webp)$", "", name, flags=re.IGNORECASE)

    # Remove augmentation suffix
    name = re.sub(r"_aug_\d+$", "", name, flags=re.IGNORECASE)

    return name


def read_classes(label_file: Path):
    classes = set()

    if not label_file.exists():
        return classes

    try:
        content = label_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()
    except Exception:
        return classes

    for line in content.splitlines():
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
    groups = defaultdict(lambda: {
        "splits": set(),
        "images": [],
        "classes": set(),
    })

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
            label_file = label_dir / f"{image.stem}.txt"

            groups[group]["splits"].add(split)
            groups[group]["images"].append(
                (split, image.name)
            )
            groups[group]["classes"].update(
                read_classes(label_file)
            )

    print("=" * 75)
    print("ORIGINAL IMAGE GROUP ANALYSIS")
    print("=" * 75)

    print(f"Total original groups: {len(groups)}")

    single_split = Counter()
    multi_split_groups = 0
    total_images = 0

    for group, info in groups.items():
        total_images += len(info["images"])

        if len(info["splits"]) == 1:
            split = next(iter(info["splits"]))
            single_split[split] += 1
        else:
            multi_split_groups += 1

    print(f"Total image files   : {total_images}")
    print()

    print("Groups by current split:")
    for split in SPLITS:
        print(f"  {split:<8}: {single_split[split]}")

    print(f"  Cross-split       : {multi_split_groups}")

    print()
    print("=" * 75)
    print("GROUP CLASS DISTRIBUTION")
    print("=" * 75)

    class_group_counts = Counter()
    multi_class_groups = 0
    no_class_groups = 0

    for group, info in groups.items():

        if not info["classes"]:
            no_class_groups += 1

        if len(info["classes"]) > 1:
            multi_class_groups += 1

        for class_id in info["classes"]:
            class_group_counts[class_id] += 1

    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"{class_id}: {class_name:<18} "
            f"{class_group_counts[class_id]} groups"
        )

    print()
    print(f"Groups with multiple classes: {multi_class_groups}")
    print(f"Groups with no classes      : {no_class_groups}")

    print()
    print("=" * 75)
    print("FIRST 30 ORIGINAL GROUPS")
    print("=" * 75)

    for index, (group, info) in enumerate(sorted(groups.items())[:30], start=1):

        class_names = [
            CLASS_NAMES[c]
            for c in sorted(info["classes"])
        ]

        print(
            f"{index:02d}. {group:<15} "
            f"images={len(info['images']):<3} "
            f"splits={','.join(sorted(info['splits'])):<20} "
            f"classes={', '.join(class_names) if class_names else 'NONE'}"
        )

    print()
    print("=" * 75)
    print("ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()