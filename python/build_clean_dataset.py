from pathlib import Path
from collections import defaultdict
import csv
import re
import shutil


SOURCE = Path("raw_dataset")
PLAN_FILE = SOURCE / "clean_split_plan.csv"
OUTPUT = Path("clean_dataset")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

CLASS_NAMES = {
    0: "Bacterial Blotch",
    1: "Dry Bubble",
    2: "Healthy",
    3: "Trichoderma",
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


def validate_label_file(label_file: Path):
    """
    Returns:
        valid, reason

    Empty label files are considered valid background images.
    """

    if not label_file.exists():
        return False, "missing label"

    text = label_file.read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    # Background image with no annotations
    if not text:
        return True, "empty label"

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        parts = line.split()

        # Segmentation format:
        # class_id x1 y1 x2 y2 ...
        if len(parts) < 7:
            return (
                False,
                f"too few values on line {line_number}",
            )

        try:
            class_id = int(float(parts[0]))
            coordinates = [
                float(value)
                for value in parts[1:]
            ]
        except ValueError:
            return (
                False,
                f"non-numeric value on line {line_number}",
            )

        if class_id not in CLASS_NAMES:
            return (
                False,
                f"invalid class {class_id}",
            )

        if len(coordinates) % 2 != 0:
            return (
                False,
                f"odd coordinate count on line {line_number}",
            )

        if any(
            value < 0 or value > 1
            for value in coordinates
        ):
            return (
                False,
                f"coordinate outside [0,1] on line {line_number}",
            )

    return True, "valid"


def load_plan():
    plan = {}

    with PLAN_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            plan[row["group"]] = row["split"]

    return plan


def collect_images():
    """
    Collect every image from the original train/valid/test folders
    and group them by original source group.
    """

    groups = defaultdict(list)

    for original_split in [
        "train",
        "valid",
        "test",
    ]:

        image_dir = (
            SOURCE
            / original_split
            / "images"
        )

        label_dir = (
            SOURCE
            / original_split
            / "labels"
        )

        if not image_dir.exists():
            continue

        for image in image_dir.iterdir():

            if not image.is_file():
                continue

            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            group = get_group_name(
                image.name
            )

            label = (
                label_dir
                / f"{image.stem}.txt"
            )

            groups[group].append(
                {
                    "image": image,
                    "label": label,
                    "original_split": original_split,
                }
            )

    return groups


def main():

    print("=" * 75)
    print("BUILDING CLEAN GROUP-AWARE DATASET")
    print("=" * 75)

    if not PLAN_FILE.exists():
        raise FileNotFoundError(
            f"Split plan not found: {PLAN_FILE}"
        )

    plan = load_plan()

    print(
        f"Groups in split plan: {len(plan)}"
    )

    groups = collect_images()

    print(
        f"Groups collected from source: {len(groups)}"
    )

    # Create output directories
    for split in [
        "train",
        "valid",
        "test",
    ]:

        (OUTPUT / split / "images").mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUTPUT / split / "labels").mkdir(
            parents=True,
            exist_ok=True,
        )

    copied = defaultdict(int)
    skipped = []
    class_counts = defaultdict(int)

    # Track destination names to detect collisions
    destination_names = defaultdict(set)

    for group, files in sorted(groups.items()):

        if group not in plan:
            print(
                f"WARNING: group missing from plan: {group}"
            )
            continue

        target_split = plan[group]

        target_image_dir = (
            OUTPUT
            / target_split
            / "images"
        )

        target_label_dir = (
            OUTPUT
            / target_split
            / "labels"
        )

        for item in files:

            source_image = item["image"]
            source_label = item["label"]

            valid, reason = validate_label_file(
                source_label
            )

            if not valid:
                skipped.append(
                    (
                        target_split,
                        source_image.name,
                        reason,
                    )
                )
                continue

            destination_name = source_image.name

            if destination_name in destination_names[
                target_split
            ]:
                print(
                    "WARNING: duplicate filename "
                    f"in {target_split}: "
                    f"{destination_name}"
                )

            destination_names[
                target_split
            ].add(destination_name)

            destination_image = (
                target_image_dir
                / destination_name
            )

            destination_label = (
                target_label_dir
                / f"{source_image.stem}.txt"
            )

            shutil.copy2(
                source_image,
                destination_image,
            )

            shutil.copy2(
                source_label,
                destination_label,
            )

            copied[target_split] += 1

            # Count classes for valid annotations
            text = source_label.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            if text:

                for line in text.splitlines():

                    parts = line.split()

                    if not parts:
                        continue

                    class_id = int(
                        float(parts[0])
                    )

                    if class_id in CLASS_NAMES:
                        class_counts[
                            (
                                target_split,
                                class_id,
                            )
                        ] += 1

    print("\n" + "=" * 75)
    print("CLEAN DATASET CREATED")
    print("=" * 75)

    for split in [
        "train",
        "valid",
        "test",
    ]:

        print(
            f"\n{split.upper()}"
        )

        print(
            f"Images copied: "
            f"{copied[split]}"
        )

        print("Instances:")

        for class_id, class_name in CLASS_NAMES.items():

            count = class_counts[
                (
                    split,
                    class_id,
                )
            ]

            print(
                f"  {class_id}: "
                f"{class_name:<18} "
                f"{count}"
            )

    print(
        f"\nSkipped invalid samples: "
        f"{len(skipped)}"
    )

    if skipped:

        print(
            "\nSkipped samples:"
        )

        for split, filename, reason in skipped[:20]:

            print(
                f"  [{split}] "
                f"{filename} -> {reason}"
            )

    print("\nOutput:")
    print(OUTPUT.resolve())

    print(
        "\nOriginal dataset has NOT been modified."
    )


if __name__ == "__main__":
    main()