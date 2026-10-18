from pathlib import Path
from collections import defaultdict, Counter
import random
import re


DATASET = Path("raw_dataset")

SPLITS = {
    "train": 0.80,
    "valid": 0.10,
    "test": 0.10,
}

CLASS_NAMES = {
    0: "Bacterial Blotch",
    1: "Dry Bubble",
    2: "Healthy",
    3: "Trichoderma",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_group_name(filename: str) -> str:
    """
    Convert an augmented Roboflow filename to its original group.

    Examples:
        img_17_aug_1_jpg.rf.hash.jpg -> img_17
        img_17_aug_2_jpg.rf.hash.jpg -> img_17
        img_17_jpg.rf.hash.jpg        -> img_17
    """

    name = Path(filename).stem

    # Remove Roboflow hash
    name = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", name)

    # Remove encoded file extension
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


def get_classes(label_file: Path) -> set[int]:
    classes = set()

    if not label_file.exists():
        return classes

    text = label_file.read_text(
        encoding="utf-8",
        errors="ignore",
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


def collect_groups():
    """
    Read all existing train/valid/test folders and combine them
    into original-image groups.
    """

    groups = defaultdict(
        lambda: {
            "images": [],
            "classes": set(),
        }
    )

    for current_split in ["train", "valid", "test"]:

        image_dir = DATASET / current_split / "images"
        label_dir = DATASET / current_split / "labels"

        if not image_dir.exists():
            continue

        for image in image_dir.iterdir():

            if not image.is_file():
                continue

            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            group = get_group_name(image.name)

            label_file = label_dir / f"{image.stem}.txt"

            groups[group]["images"].append(
                (current_split, image)
            )

            groups[group]["classes"].update(
                get_classes(label_file)
            )

    return groups


def calculate_targets(groups):
    total_groups = len(groups)

    group_targets = {}

    for split, ratio in SPLITS.items():
        group_targets[split] = round(total_groups * ratio)

    # Correct rounding so total equals total_groups
    difference = total_groups - sum(group_targets.values())

    if difference != 0:
        group_targets["train"] += difference

    class_totals = Counter()

    for info in groups.values():
        for class_id in info["classes"]:
            class_totals[class_id] += 1

    class_targets = {}

    for split, ratio in SPLITS.items():
        class_targets[split] = {}

        for class_id in CLASS_NAMES:
            class_targets[split][class_id] = round(
                class_totals[class_id] * ratio
            )

    return group_targets, class_totals, class_targets


def choose_split(
    group_classes,
    current_group_counts,
    current_class_counts,
    group_targets,
    class_targets,
):
    """
    Select the split that currently needs this group's classes most.

    This is a deterministic greedy strategy designed to balance
    class presence while keeping complete source groups together.
    """

    best_split = None
    best_score = float("-inf")

    for split in SPLITS:

        if current_group_counts[split] >= group_targets[split]:
            continue

        # How much group capacity remains?
        group_deficit = (
            group_targets[split]
            - current_group_counts[split]
        )

        score = group_deficit / max(group_targets[split], 1)

        # Give additional weight to classes that are currently
        # below their target for this split.
        for class_id in group_classes:

            target = class_targets[split][class_id]
            current = current_class_counts[split][class_id]

            if target <= 0:
                continue

            deficit = max(target - current, 0)
            score += deficit / target

        if score > best_score:
            best_score = score
            best_split = split

    return best_split


def main():

    print("=" * 75)
    print("GROUP-AWARE DATASET SPLIT PLANNER")
    print("=" * 75)

    groups = collect_groups()

    print(f"Original groups found: {len(groups)}")

    group_targets, class_totals, class_targets = calculate_targets(groups)

    print("\nTarget group counts:")
    for split in SPLITS:
        print(
            f"  {split:<6}: "
            f"{group_targets[split]} groups"
        )

    print("\nTotal groups containing each class:")
    for class_id, name in CLASS_NAMES.items():
        print(
            f"  {class_id}: {name:<18} "
            f"{class_totals[class_id]} groups"
        )

    # Start with groups containing rarer classes.
    # This helps make sure disease classes are distributed.
    group_items = list(groups.items())

    random.seed(42)
    random.shuffle(group_items)

    def rarity_score(item):
        _, info = item

        score = 0.0

        for class_id in info["classes"]:
            total = class_totals[class_id]
            score += 1.0 / max(total, 1)

        # Process larger groups first when rarity is similar.
        score += len(info["images"]) / 100000.0

        return score

    group_items.sort(
        key=rarity_score,
        reverse=True,
    )

    assignments = {}

    current_group_counts = Counter()

    current_class_counts = {
        split: Counter()
        for split in SPLITS
    }

    for group_name, info in group_items:

        split = choose_split(
            info["classes"],
            current_group_counts,
            current_class_counts,
            group_targets,
            class_targets,
        )

        if split is None:
            # Safety fallback
            available = [
                s for s in SPLITS
                if current_group_counts[s] < group_targets[s]
            ]

            if not available:
                available = list(SPLITS)

            split = min(
                available,
                key=lambda s: current_group_counts[s],
            )

        assignments[group_name] = split

        current_group_counts[split] += 1

        for class_id in info["classes"]:
            current_class_counts[split][class_id] += 1

    print("\n" + "=" * 75)
    print("PROPOSED SPLIT")
    print("=" * 75)

    for split in SPLITS:

        print(f"\n{split.upper()}")

        print(
            f"Groups : "
            f"{current_group_counts[split]}"
        )

        image_count = sum(
            len(groups[group]["images"])
            for group, assigned_split in assignments.items()
            if assigned_split == split
        )

        print(
            f"Images : "
            f"{image_count}"
        )

        print("Class-containing groups:")

        for class_id, name in CLASS_NAMES.items():
            print(
                f"  {name:<18}: "
                f"{current_class_counts[split][class_id]}"
            )

    print("\n" + "=" * 75)
    print("LEAKAGE CHECK")
    print("=" * 75)

    groups_in_each_split = defaultdict(list)

    for group, split in assignments.items():
        groups_in_each_split[split].append(group)

    overlap_found = False

    for i, split_a in enumerate(SPLITS):
        for j, split_b in enumerate(SPLITS):

            if j <= i:
                continue

            overlap = (
                set(groups_in_each_split[split_a])
                & set(groups_in_each_split[split_b])
            )

            if overlap:
                overlap_found = True
                print(
                    f"OVERLAP: {split_a} <-> {split_b}: "
                    f"{len(overlap)} groups"
                )

    if not overlap_found:
        print("No group appears in more than one split.")

    print("\n" + "=" * 75)
    print("SPLIT PLAN COMPLETED")
    print("=" * 75)

    # Save plan for later use.
    plan_file = DATASET / "clean_split_plan.csv"

    with plan_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write("group,split\n")

        for group, split in sorted(assignments.items()):
            f.write(
                f"{group},{split}\n"
            )

    print(f"\nPlan saved to: {plan_file}")


if __name__ == "__main__":
    main()