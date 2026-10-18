from pathlib import Path
from collections import Counter, defaultdict
import re


DATASET = Path("clean_dataset")

CLASS_NAMES = {
    0: "Bacterial Blotch",
    1: "Dry Bubble",
    2: "Healthy",
    3: "Trichoderma",
}

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_group_name(filename: str) -> str:
    name = Path(filename).stem

    # Remove Roboflow hash
    name = re.sub(
        r"\.rf\.[A-Za-z0-9]+$",
        "",
        name,
    )

    # Remove encoded extension
    name = re.sub(
        r"_(jpg|jpeg|png|bmp|webp)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Remove augmentation suffix
    name = re.sub(
        r"_aug_\d+$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return name


def verify_split(split: str):

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = [
        p for p in label_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    ]

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    extra_labels = label_stems - image_stems

    class_counts = Counter()

    empty_labels = 0
    invalid_annotations = 0
    total_instances = 0

    for label in labels:

        text = label.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if not text:
            empty_labels += 1
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):

            parts = line.split()

            if len(parts) < 7:
                invalid_annotations += 1
                continue

            try:
                class_id = int(float(parts[0]))
                coords = [
                    float(x)
                    for x in parts[1:]
                ]
            except ValueError:
                invalid_annotations += 1
                continue

            if class_id not in CLASS_NAMES:
                invalid_annotations += 1
                continue

            if len(coords) % 2 != 0:
                invalid_annotations += 1
                continue

            if any(
                x < 0 or x > 1
                for x in coords
            ):
                invalid_annotations += 1
                continue

            class_counts[class_id] += 1
            total_instances += 1

    print("\n" + "=" * 70)
    print(f"{split.upper()} CLEAN DATASET")
    print("=" * 70)

    print(f"Images              : {len(images)}")
    print(f"Label files         : {len(labels)}")
    print(f"Missing labels      : {len(missing_labels)}")
    print(f"Extra labels        : {len(extra_labels)}")
    print(f"Empty label files   : {empty_labels}")
    print(f"Total instances     : {total_instances}")
    print(f"Invalid annotations : {invalid_annotations}")

    print("\nClass instances:")

    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"  {class_id}: "
            f"{class_name:<18} "
            f"{class_counts[class_id]}"
        )

    return images, missing_labels, extra_labels


def main():

    print("=" * 70)
    print("CLEAN DATASET VERIFICATION")
    print("=" * 70)

    all_groups = defaultdict(set)

    for split in SPLITS:

        images, _, _ = verify_split(split)

        for image in images:
            group = get_group_name(image.name)
            all_groups[group].add(split)

    cross_split = {
        group: splits
        for group, splits in all_groups.items()
        if len(splits) > 1
    }

    print("\n" + "=" * 70)
    print("GROUP LEAKAGE CHECK")
    print("=" * 70)

    print(
        f"Original groups found: {len(all_groups)}"
    )

    print(
        f"Groups appearing in multiple splits: "
        f"{len(cross_split)}"
    )

    if cross_split:
        print("\nFirst 20 overlapping groups:")

        for group, splits in list(
            sorted(cross_split.items())
        )[:20]:

            print(
                f"  {group} -> "
                f"{', '.join(sorted(splits))}"
            )

    else:
        print("NO GROUP LEAKAGE FOUND ✅")

    print("\n" + "=" * 70)
    print("VERIFICATION FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()