from pathlib import Path
import time

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "segment"
    / "runs"
    / "mushroom_training_v2"
    / "yolov8n_seg_v2"
    / "weights"
    / "best.pt"
)
PREDICTION_DIR = Path("runs/predictions")
WEBCAM_DIR = Path("runs/webcam")
CONFIDENCE_THRESHOLD = 0.60
MIN_MASK_AREA_RATIO = 0.005
MUSHROOM_CLASS_IDS = {0, 1, 2, 3}


def valid_detections_from(result, frame):
    """Return only detections that pass class, confidence, and mask-area filters."""
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return []

    frame_area = frame.shape[0] * frame.shape[1]
    valid_detections = []

    for index in range(len(result.boxes)):
        class_id = int(result.boxes.cls[index].item())
        if class_id not in MUSHROOM_CLASS_IDS:
            continue

        confidence = float(result.boxes.conf[index].item())
        if not confidence >= CONFIDENCE_THRESHOLD:
            continue

        mask = result.masks.data[index].detach().cpu().numpy()
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = mask > 0.5
        mask_area = float(mask.sum())
        mask_area_ratio = mask_area / frame_area
        if mask_area_ratio < MIN_MASK_AREA_RATIO:
            continue

        valid_detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": confidence,
                "mask_area": mask_area,
                "mask": mask,
                "box": result.boxes.xyxy[index].detach().cpu().numpy(),
            }
        )

    return valid_detections


def diagnosis_from(detections):
    """Use the highest-confidence valid detection for the overall diagnosis."""
    if not detections:
        return None, "MUSHROOM NOT DETECTED", 0.0

    best = max(detections, key=lambda item: item["confidence"])
    diagnosis = best["class_name"]
    status = (
        "HEALTHY MUSHROOM"
        if diagnosis.lower() == "healthy"
        else "DISEASE DETECTED"
    )
    return diagnosis, status, best["confidence"]


def annotate_valid_detections(frame, valid_detections):
    """Draw masks, polygons, boxes, and labels only for validated detections."""
    annotated = frame.copy()
    colors = {
        0: (0, 165, 255),
        1: (255, 0, 255),
        2: (0, 200, 0),
        3: (0, 0, 255),
    }
    mask_overlay = frame.copy()

    for detection in valid_detections:
        color = colors[detection["class_id"]]
        mask = detection["mask"]
        mask_overlay[mask] = color

    annotated = cv2.addWeighted(mask_overlay, 0.4, annotated, 0.6, 0)

    for detection in valid_detections:
        color = colors[detection["class_id"]]
        mask = detection["mask"].astype("uint8")
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, color, 2)

        x1, y1, x2, y2 = (
            int(round(float(value))) for value in detection["box"]
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )
        label_y = max(y1 - 8, 20)
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def predict_frame(model, frame):
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    valid_detections = valid_detections_from(results[0], frame)
    return annotate_valid_detections(frame, valid_detections), valid_detections


def draw_overlay(image, valid_detections, fps=None):
    """Add a readable demonstration overlay to an annotated image."""
    height, width = image.shape[:2]
    diagnosis, status, confidence = diagnosis_from(valid_detections)
    no_mushroom = not valid_detections
    object_count = len(valid_detections)
    panel_bottom = 105 if no_mushroom else (214 if fps is not None else 182)
    cv2.rectangle(image, (10, 10), (min(500, width - 10), panel_bottom), (0, 0, 0), -1)

    if no_mushroom:
        lines = [("MUSHROOM NOT DETECTED", 0.68, (0, 0, 255))]
        if fps is not None:
            lines.append((f"FPS: {fps:.1f}", 0.58, (255, 255, 255)))
    else:
        lines = [
            ("MUSHROOM DETECTED", 0.64, (0, 255, 0) if status == "HEALTHY MUSHROOM" else (0, 0, 255)),
            (f"Status: {status}", 0.60, (255, 255, 255)),
            (f"Class: {diagnosis}", 0.62, (255, 255, 255)),
            (f"Confidence: {confidence * 100:.1f}%", 0.58, (255, 255, 255)),
            (f"Objects: {object_count}", 0.58, (255, 255, 255)),
        ]
    if fps is not None and not no_mushroom:
        lines.append((f"FPS: {fps:.1f}", 0.58, (255, 255, 255)))

    y = 38
    for text, scale, color in lines:
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += 31

    cv2.putText(
        image,
        "Q: Quit    S: Save",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def print_image_result(valid_detections, output_path):
    diagnosis, status, confidence = diagnosis_from(valid_detections)
    print("\n" + "=" * 64)
    print("OYSTER MUSHROOM AI DETECTION")
    print("=" * 64)
    print("Mode: IMAGE\n")

    if valid_detections:
        print("Detections:")
        for number, detection in enumerate(valid_detections, start=1):
            print(
                f"  {number}. {detection['class_name']}: "
                f"{detection['confidence'] * 100:.2f}% "
                f"(mask area: {detection['mask_area']:.0f} px)"
            )
        print()
        print("MUSHROOM DETECTED")
        print(f"Class           : {diagnosis}")
        print(f"Confidence      : {confidence * 100:.2f}%")
        print(f"Objects         : {len(valid_detections)}")
        print(f"Status          : {status}")
    else:
        print("MUSHROOM NOT DETECTED")
    print(f"\nAnnotated image:\n{output_path.resolve()}")
    print("=" * 64)


def image_mode(model):
    image_input = input("\nEnter the full path of the mushroom image: ").strip().strip('"')
    image_path = Path(image_input)
    if not image_path.is_file():
        print(f"ERROR: Image not found: {image_path}")
        return

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    annotated, valid_detections = predict_frame(model, frame)
    annotated = draw_overlay(annotated, valid_detections)

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTION_DIR / f"{image_path.stem}_app_prediction.jpg"
    cv2.imwrite(str(output_path), annotated)
    print_image_result(valid_detections, output_path)

    cv2.imshow("Oyster Mushroom Image Result", annotated)
    print("Press any key in the image window to return to the menu.")
    cv2.waitKey(0)
    cv2.destroyWindow("Oyster Mushroom Image Result")


def webcam_mode(model):
    print("\nOpening webcam index 0...")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam index 0.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    WEBCAM_DIR.mkdir(parents=True, exist_ok=True)

    print("Webcam started. Controls: Q = Quit, S = Save annotated frame")
    previous_time = time.time()
    saved_counter = 0

    while True:
        success, frame = camera.read()
        if not success:
            print("ERROR: Failed to read webcam frame.")
            break

        annotated, valid_detections = predict_frame(model, frame)
        diagnosis, _, confidence = diagnosis_from(valid_detections)

        now = time.time()
        fps = 1.0 / (now - previous_time) if now > previous_time else 0.0
        previous_time = now
        annotated = draw_overlay(
            annotated, valid_detections, fps
        )

        cv2.imshow("Oyster Mushroom Live Monitor", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            saved_counter += 1
            output_path = WEBCAM_DIR / f"app_webcam_capture_{saved_counter:03d}.jpg"
            cv2.imwrite(str(output_path), annotated)
            if diagnosis is None:
                print(f"Saved: {output_path.resolve()} | MUSHROOM NOT DETECTED")
            else:
                print(
                    f"Saved: {output_path.resolve()} | {diagnosis} "
                    f"({confidence * 100:.1f}%), FPS {fps:.1f}"
                )

    camera.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading trained YOLOv8 segmentation model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully.")

    while True:
        print("\n" + "=" * 64)
        print("OYSTER MUSHROOM AI DETECTION")
        print("=" * 64)
        print("Select Mode:")
        print("[1] Upload Image")
        print("[2] Live Webcam")
        print("[Q] Quit")

        choice = input("\nChoose an option: ").strip().lower()
        if choice == "1":
            image_mode(model)
        elif choice == "2":
            webcam_mode(model)
        elif choice == "q":
            break
        else:
            print("Invalid option. Choose 1, 2, or Q.")

    cv2.destroyAllWindows()
    print("\nApplication closed.")


if __name__ == "__main__":
    main()
