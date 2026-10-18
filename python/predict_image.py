from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path(
    r"runs\segment\runs\mushroom_training_v2\yolov8n_seg_v2\weights\best.pt"
)
OUTPUT_DIR = Path("runs/predictions")


def main():
    print("=" * 70)
    print("OYSTER MUSHROOM SINGLE IMAGE PREDICTION")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    image_path = Path(
        input("\nEnter the full path of the mushroom image: ").strip().strip('"')
    )
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading trained model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded.")

    print("\nRunning prediction...")
    results = model.predict(
        source=str(image_path),
        imgsz=512,
        device=0,
        conf=0.25,
        save=False,
        verbose=False,
    )

    result = results[0]
    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    has_detections = result.boxes is not None and len(result.boxes) > 0
    if has_detections:
        names = result.names
        for i in range(len(result.boxes)):
            class_id = int(result.boxes.cls[i].item())
            confidence = float(result.boxes.conf[i].item())
            print(
                f"Object {i + 1}: "
                f"{names[class_id]} "
                f"({confidence * 100:.2f}%)"
            )
    else:
        print("No mushroom/disease object detected.")

    mask_view = result.plot(boxes=False, labels=False, masks=True)
    annotated = result.plot()
    output_path = OUTPUT_DIR / f"{image_path.stem}_prediction.jpg"
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f"Could not save annotated image: {output_path}")

    print("\nAnnotated result saved to:")
    print(output_path.resolve())
    print("\nDisplaying the original image, segmentation mask, and annotated result.")
    print("Press any key while an image window is focused to close the displays.")

    cv2.imshow("Original image", original)
    cv2.imshow("Segmentation mask/polygon", mask_view)
    cv2.imshow("Annotated prediction", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
