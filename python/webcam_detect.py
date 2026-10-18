from pathlib import Path
import time

import cv2
from ultralytics import YOLO


MODEL_PATH = Path(
    r"runs\segment\runs\mushroom_training_v2\yolov8n_seg_v2\weights\best.pt"
)

OUTPUT_DIR = Path("runs/webcam")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():
    print("=" * 70)
    print("OYSTER MUSHROOM LIVE WEBCAM DETECTION")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("\nLoading YOLOv8 model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    print("\nOpening webcam...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam index 0.")
        print("\nTrying to identify available camera indexes...")

        for index in range(1, 5):
            test_camera = cv2.VideoCapture(index)

            if test_camera.isOpened():
                print(
                    f"Camera available at index {index}"
                )
                test_camera.release()

            else:
                test_camera.release()

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print("\nWebcam started.")
    print("Controls:")
    print("  Q = Quit")
    print("  S = Save annotated frame")

    previous_time = time.time()
    saved_counter = 0

    while True:
        success, frame = camera.read()

        if not success:
            print(
                "ERROR: Failed to read webcam frame."
            )
            break

        results = model.predict(
            source=frame,
            imgsz=512,
            device=0,
            conf=0.25,
            verbose=False
        )

        result = results[0]
        annotated = result.plot()
        detected = []

        if result.boxes is not None:
            for i in range(
                len(result.boxes)
            ):
                class_id = int(
                    result.boxes.cls[i].item()
                )
                confidence = float(
                    result.boxes.conf[i].item()
                )
                class_name = result.names[
                    class_id
                ]
                detected.append(
                    {
                        "class": class_name,
                        "confidence": confidence
                    }
                )

        if not detected:
            overall_status = "NO DETECTION"
            overall_confidence = 0.0

        else:
            best_detection = max(
                detected,
                key=lambda item: item["confidence"]
            )
            overall_class = best_detection[
                "class"
            ]
            overall_confidence = best_detection[
                "confidence"
            ]

            if overall_class.lower() == "healthy":
                overall_status = "HEALTHY"
            else:
                overall_status = "DISEASE DETECTED"

        current_time = time.time()
        elapsed = (
            current_time -
            previous_time
        )
        fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0.0
        )
        previous_time = current_time

        cv2.rectangle(
            annotated,
            (10, 10),
            (420, 145),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            annotated,
            "OYSTER MUSHROOM MONITOR",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            f"Status: {overall_status}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if detected:
            cv2.putText(
                annotated,
                f"Class: {overall_class}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated,
                f"Confidence: {overall_confidence * 100:.1f}%",
                (20, 127),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.putText(
            annotated,
            f"Objects: {len(detected)}",
            (annotated.shape[1] - 180, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (annotated.shape[1] - 180, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            "Q: Quit    S: Save",
            (20, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Oyster Mushroom Disease Detection",
            annotated
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            saved_counter += 1
            output_path = (
                OUTPUT_DIR /
                f"webcam_capture_{saved_counter:03d}.jpg"
            )
            cv2.imwrite(
                str(output_path),
                annotated
            )
            print(
                f"Saved: {output_path.resolve()}"
            )

    camera.release()
    cv2.destroyAllWindows()

    print("\nWebcam stopped.")
    print("=" * 70)


if __name__ == "__main__":
    main()
