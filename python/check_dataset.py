from pathlib import Path
from collections import Counter


DATASET = Path("raw_dataset")

SPLITS = {
    "train": DATASET / "train",
    "valid": DATASET / "valid",
    "test": DATASET / "test",
}

CLASS_NAMES = {
    0: "Bacterial Blotch",
    1: "Dry Bubble",
    2: "Healthy",
    3: "Trichoderma",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_split(split_name: str, split_path: Path) -> None:
    image_dir = split_path / "images"
    label_dir = split_path / "labels"

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} DATASET")
    print("=" * 60)

    if not image_dir.exists():
        print(f"ERROR: Image folder not found: {image_dir}")
        return

    if not label_dir.exists():
        print(f"ERROR: Label folder not found: {label_dir}")
        return

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = [
        p for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    extra_labels = label_stems - image_stems

    class_counts = Counter()
    invalid_lines = []
    empty_labels = 0
    total_instances = 0

    for label_file in labels:
        try:
            content = label_file.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = label_file.read_text(encoding="utf-8", errors="replace").strip()

        if not content:
            empty_labels += 1
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            parts = line.split()

            # Segmentation annotation:
            # class_id x1 y1 x2 y2 x3 y3 ...
            if len(parts) < 7:
                invalid_lines.append(
                    f"{label_file.name}: line {line_number} -> too few values"
                )
                continue

            try:
                class_id = int(float(parts[0]))
                coordinates = [float(x) for x in parts[1:]]
            except ValueError:
                invalid_lines.append(
                    f"{label_file.name}: line {line_number} -> non-numeric value"
                )
                continue

            if class_id not in CLASS_NAMES:
                invalid_lines.append(
                    f"{label_file.name}: line {line_number} -> invalid class {class_id}"
                )
                continue

            # Polygon needs x,y pairs
            if len(coordinates) % 2 != 0:
                invalid_lines.append(
                    f"{label_file.name}: line {line_number} -> odd coordinate count"
                )
                continue

            # YOLO normalized coordinates should normally be in [0, 1]
            if any(value < 0 or value > 1 for value in coordinates):
                invalid_lines.append(
                    f"{label_file.name}: line {line_number} -> coordinate outside [0,1]"
                )
                continue

            class_counts[class_id] += 1
            total_instances += 1

    print(f"Images              : {len(images)}")
    print(f"Label files          : {len(labels)}")
    print(f"Missing labels       : {len(missing_labels)}")
    print(f"Extra labels         : {len(extra_labels)}")
    print(f"Empty label files    : {empty_labels}")
    print(f"Total instances      : {total_instances}")
    print(f"Invalid annotations : {len(invalid_lines)}")

    print("\nClass distribution:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name:<18} {class_counts[class_id]}")

    if missing_labels:
        print("\nFirst missing labels:")
        for name in sorted(missing_labels)[:10]:
            print(f"  {name}")

    if extra_labels:
        print("\nFirst extra labels:")
        for name in sorted(extra_labels)[:10]:
            print(f"  {name}")

    if invalid_lines:
        print("\nFirst invalid annotations:")
        for item in invalid_lines[:10]:
            print(f"  {item}")


def main() -> None:
    print("YOLOv8 OYSTER FUNGI DATASET CHECK")
    print("=" * 60)

    for split_name, split_path in SPLITS.items():
        check_split(split_name, split_path)

    print("\n" + "=" * 60)
    print("DATASET CHECK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()