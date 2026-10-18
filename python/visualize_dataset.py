from pathlib import Path
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path("clean_dataset")

CLASS_NAMES = {
    0: "Bacterial Blotch",
    1: "Dry Bubble",
    2: "Healthy",
    3: "Trichoderma",
}

# Display colors in RGB format
COLORS = {
    0: (255, 0, 0),       # Red
    1: (0, 255, 0),       # Green
    2: (0, 0, 255),       # Blue
    3: (255, 165, 0),     # Orange
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SAMPLES_PER_VIEW = 6


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(split: str):
    """
    Find all supported images inside:
        clean_dataset/<split>/images
    """

    image_dir = DATASET / split / "images"

    if not image_dir.exists():
        print(f"WARNING: Folder not found: {image_dir}")
        return []

    return sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


# ============================================================
# READ YOLO SEGMENTATION LABEL
# ============================================================

def read_label(label_file: Path):
    """
    Read a YOLO segmentation label.

    Format:
        class_id x1 y1 x2 y2 x3 y3 ...

    Coordinates are normalized between 0 and 1.
    """

    objects = []

    if not label_file.exists():
        return objects

    try:
        text = label_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()
    except Exception as exc:
        print(f"Could not read {label_file}: {exc}")
        return objects

    # Empty label = no annotation
    if not text:
        return objects

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        parts = line.split()

        # A valid segmentation annotation needs:
        # class_id + at least 3 x/y points = 7 values
        if len(parts) < 7:
            print(
                f"Skipping invalid annotation: "
                f"{label_file.name}, line {line_number}"
            )
            continue

        try:
            class_id = int(float(parts[0]))

            coordinates = [
                float(value)
                for value in parts[1:]
            ]

        except ValueError:
            print(
                f"Non-numeric annotation: "
                f"{label_file.name}, line {line_number}"
            )
            continue

        if class_id not in CLASS_NAMES:
            print(
                f"Unknown class {class_id}: "
                f"{label_file.name}, line {line_number}"
            )
            continue

        # Need x,y pairs
        if len(coordinates) % 2 != 0:
            print(
                f"Odd number of coordinates: "
                f"{label_file.name}, line {line_number}"
            )
            continue

        # Coordinates must be normalized
        if any(
            coordinate < 0 or coordinate > 1
            for coordinate in coordinates
        ):
            print(
                f"Coordinates outside [0,1]: "
                f"{label_file.name}, line {line_number}"
            )
            continue

        points = []

        for i in range(0, len(coordinates), 2):

            x = coordinates[i]
            y = coordinates[i + 1]

            points.append(
                (x, y)
            )

        objects.append(
            {
                "class_id": class_id,
                "points": points,
            }
        )

    return objects


# ============================================================
# DRAW SEGMENTATION
# ============================================================

def draw_annotations(image_path: Path):
    """
    Load an image and draw all YOLO segmentation polygons.
    """

    # Read image
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            f"ERROR: Could not read image: "
            f"{image_path}"
        )
        return None, []

    # OpenCV BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width = image.shape[:2]

    # Corresponding label
    label_path = (
        image_path.parent.parent
        / "labels"
        / f"{image_path.stem}.txt"
    )

    objects = read_label(
        label_path
    )

    # Draw every annotated object
    for obj in objects:

        class_id = obj["class_id"]
        points = obj["points"]

        if len(points) < 3:
            continue

        polygon_points = []

        for x, y in points:

            pixel_x = int(
                round(x * width)
            )

            pixel_y = int(
                round(y * height)
            )

            # Keep coordinates safely inside image
            pixel_x = max(
                0,
                min(pixel_x, width - 1)
            )

            pixel_y = max(
                0,
                min(pixel_y, height - 1)
            )

            polygon_points.append(
                [pixel_x, pixel_y]
            )

        # ====================================================
        # IMPORTANT FIX FOR OPENCV 5
        # ====================================================

        polygon = np.array(
            polygon_points,
            dtype=np.int32
        ).reshape(
            (-1, 1, 2)
        )

        color = COLORS[class_id]

        # ----------------------------------------------------
        # Draw transparent filled mask
        # ----------------------------------------------------

        overlay = image.copy()

        cv2.fillPoly(
            overlay,
            [polygon],
            color
        )

        image = cv2.addWeighted(
            overlay,
            0.25,
            image,
            0.75,
            0
        )

        # ----------------------------------------------------
        # Draw polygon boundary
        # ----------------------------------------------------

        cv2.polylines(
            image,
            [polygon],
            isClosed=True,
            color=color,
            thickness=2
        )

        # ----------------------------------------------------
        # Label position
        # ----------------------------------------------------

        label_x = int(
            polygon[0][0][0]
        )

        label_y = int(
            polygon[0][0][1]
        )

        label_text = CLASS_NAMES[
            class_id
        ]

        cv2.putText(
            image,
            label_text,
            (
                label_x,
                max(
                    label_y - 8,
                    25
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA
        )

    return image, objects


# ============================================================
# SHOW SAMPLE IMAGES
# ============================================================

def show_samples(
    split: str,
    samples,
    title: str,
    output_filename: str
):
    """
    Display up to six sample images in a 2x3 grid.
    """

    if not samples:
        print(
            f"No images available for: "
            f"{split} / {title}"
        )
        return

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    axes = axes.flatten()

    # Turn all axes off initially
    for axis in axes:
        axis.axis("off")

    for axis, image_path in zip(
        axes,
        samples
    ):

        image, objects = draw_annotations(
            image_path
        )

        if image is None:
            continue

        axis.imshow(image)

        # Determine classes present
        class_names = sorted(
            {
                CLASS_NAMES[
                    obj["class_id"]
                ]
                for obj in objects
            }
        )

        if class_names:

            axis.set_title(
                ", ".join(class_names),
                fontsize=10
            )

        else:

            axis.set_title(
                "EMPTY LABEL",
                fontsize=10
            )

        axis.axis("off")

    # Hide unused axes
    for axis in axes[len(samples):]:
        axis.axis("off")

    figure.suptitle(
        f"{split.upper()} - {title}",
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        DATASET.parent
        / output_filename
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Saved visualization:"
    )

    print(
        output_path.resolve()
    )

    # Display window
    plt.show()

    # Close figure after window is closed
    plt.close(
        figure
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLEAN DATASET VISUAL INSPECTION")
    print("=" * 70)

    random.seed(42)

    # --------------------------------------------------------
    # Check dataset directory
    # --------------------------------------------------------

    if not DATASET.exists():

        print(
            f"ERROR: Dataset folder not found:"
        )

        print(
            DATASET.resolve()
        )

        return

    # --------------------------------------------------------
    # Process each split
    # --------------------------------------------------------

    for split in [
        "train",
        "valid",
        "test"
    ]:

        images = find_images(
            split
        )

        print(
            f"\n{split.upper()}: "
            f"{len(images)} images"
        )

        # ====================================================
        # FIND ANNOTATED IMAGES
        # ====================================================

        annotated_images = []

        for image in images:

            label_path = (
                DATASET
                / split
                / "labels"
                / f"{image.stem}.txt"
            )

            objects = read_label(
                label_path
            )

            if objects:
                annotated_images.append(
                    image
                )

        # Random annotated samples
        annotated_samples = random.sample(
            annotated_images,
            min(
                SAMPLES_PER_VIEW,
                len(annotated_images)
            )
        )

        # ====================================================
        # SHOW ANNOTATED SAMPLES
        # ====================================================

        show_samples(
            split=split,
            samples=annotated_samples,
            title="Annotated Samples",
            output_filename=(
                f"dataset_check_"
                f"{split}_annotated.png"
            )
        )

        # ====================================================
        # FIND EMPTY LABEL IMAGES
        # ====================================================

        empty_images = []

        for image in images:

            label_path = (
                DATASET
                / split
                / "labels"
                / f"{image.stem}.txt"
            )

            if not label_path.exists():
                continue

            text = label_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if not text:
                empty_images.append(
                    image
                )

        # Random empty-label samples
        empty_samples = random.sample(
            empty_images,
            min(
                SAMPLES_PER_VIEW,
                len(empty_images)
            )
        )

        # ====================================================
        # SHOW EMPTY LABEL SAMPLES
        # ====================================================

        if empty_samples:

            show_samples(
                split=split,
                samples=empty_samples,
                title="Empty Label Samples",
                output_filename=(
                    f"dataset_check_"
                    f"{split}_empty.png"
                )
            )

        else:

            print(
                f"No empty-label images in "
                f"{split}."
            )

    print("\n" + "=" * 70)
    print("VISUAL INSPECTION COMPLETED")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()