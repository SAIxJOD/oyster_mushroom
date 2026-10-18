from pathlib import Path
from collections import Counter


DATASET = Path("raw_dataset")

CLASS_NAMES = {
    0: "Bacterial Blotch",
    1: "Dry Bubble",
    2: "Healthy",
    3: "Trichoderma",
}

SPLITS = ["train", "valid", "test"]


def analyze_split(split_name: str):
    image_dir = DATASET / split_name / "images"
    label_dir = DATASET / split_name / "labels"

    print("\n" + "=" * 65)
    print(f"{split_name.upper()} - IMAGE LEVEL ANALYSIS")
    print("=" * 65)

    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    class_image_counts = Counter()
    no_label_images = 0
    multi_class_images = 0
    total_images = len(image_files)

    for image in image_files:
        label_file = label_dir / f"{image.stem}.txt"

        if not label_file.exists():
            no_label_images += 1
            continue

        text = label_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if not text:
            no_label_images += 1
            continue

        classes_in_image = set()

        for line in text.splitlines():
            parts = line.split()

            if not parts:
                continue

            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue

            if class_id in CLASS_NAMES:
                classes_in_image.add(class_id)

        if not classes_in_image:
            no_label_images += 1
            continue

        if len(classes_in_image) > 1:
            multi_class_images += 1

        for class_id in classes_in_image:
            class_image_counts[class_id] += 1

    print(f"Total images             : {total_images}")
    print(f"Images with no labels    : {no_label_images}")
    print(f"Images with multiple classes: {multi_class_images}")

    print("\nImages containing each class:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {class_id}: {name:<18} {class_image_counts[class_id]}")


def main():
    print("YOLOv8 OYSTER FUNGI - IMAGE LEVEL CLASS ANALYSIS")

    for split in SPLITS:
        analyze_split(split)

    print("\n" + "=" * 65)
    print("IMAGE LEVEL ANALYSIS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()