from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import cv2


MODEL_PATH = Path(
    r"runs\segment\runs\mushroom_training_v2\yolov8n_seg_v2\weights\best.pt"
)

OUTPUT_DIR = Path("runs/predictions")


def main():
    print("=" * 70)
    print("OYSTER MUSHROOM AI DISEASE DETECTION")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    image_input = input(
        "\nEnter the full path of the mushroom image: "
    ).strip().strip('"')

    image_path = Path(image_input)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading trained YOLOv8 model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    print("\nRunning inference...")

    results = model.predict(
        source=str(image_path),
        imgsz=512,
        device=0,
        conf=0.25,
        save=False,
        verbose=False
    )

    result = results[0]

    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)

    if result.boxes is None or len(result.boxes) == 0:
        print("\nNo mushroom/disease region detected.")
        print("Status: NO DETECTION")

        annotated = result.plot()

        output_path = (
            OUTPUT_DIR /
            f"{image_path.stem}_prediction.jpg"
        )

        cv2.imwrite(
            str(output_path),
            annotated
        )

        print(
            f"\nAnnotated image saved to:\n"
            f"{output_path.resolve()}"
        )

        return

    names = result.names

    detected_classes = []
    detected_confidences = []

    print()

    for i in range(len(result.boxes)):
        class_id = int(
            result.boxes.cls[i].item()
        )

        confidence = float(
            result.boxes.conf[i].item()
        )

        class_name = names[class_id]

        detected_classes.append(
            class_name
        )

        detected_confidences.append(
            confidence
        )

        print(
            f"Object {i + 1}: "
            f"{class_name:<18} "
            f"{confidence * 100:6.2f}%"
        )

    class_counter = Counter(
        detected_classes
    )

    max_count = max(
        class_counter.values()
    )

    candidates = [
        class_name
        for class_name, count
        in class_counter.items()
        if count == max_count
    ]

    if len(candidates) > 1:
        candidate_confidences = {}

        for candidate in candidates:
            candidate_confidences[candidate] = max(
                confidence
                for class_name, confidence
                in zip(
                    detected_classes,
                    detected_confidences
                )
                if class_name == candidate
            )

        dominant_class = max(
            candidate_confidences,
            key=candidate_confidences.get
        )

    else:
        dominant_class = candidates[0]

    dominant_confidence = max(
        confidence
        for class_name, confidence
        in zip(
            detected_classes,
            detected_confidences
        )
        if class_name == dominant_class
    )

    if dominant_class.lower() == "healthy":
        status = "HEALTHY"
    else:
        status = "DISEASE DETECTED"

    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS")
    print("=" * 70)

    print(
        f"\nDiagnosis       : {dominant_class}"
    )

    print(
        f"Confidence      : "
        f"{dominant_confidence * 100:.2f}%"
    )

    print(
        f"Detected regions: "
        f"{len(detected_classes)}"
    )

    print(
        f"Status          : {status}"
    )

    print("\n" + "=" * 70)

    annotated = result.plot()

    output_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_review_prediction.jpg"
    )

    cv2.imwrite(
        str(output_path),
        annotated
    )

    print(
        f"\nAnnotated result saved to:\n"
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
