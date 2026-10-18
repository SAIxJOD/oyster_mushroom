from pathlib import Path
from ultralytics import YOLO

CHECKPOINT = Path(
    r"C:\Users\saina\Desktop\mushroom_project\runs\segment\runs\mushroom_training_v2\yolov8n_seg_v2\weights\last.pt"
)


def main():
    print("=" * 75)
    print("OYSTER MUSHROOM DISEASE DETECTION")
    print("YOLOv8n INSTANCE SEGMENTATION TRAINING - V2 RESUME")
    print("=" * 75)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {CHECKPOINT}")

    print(f"\nLoading existing V2 checkpoint: {CHECKPOINT}")
    model = YOLO(str(CHECKPOINT))
    print("Resuming training from checkpoint state...")
    model.train(resume=True)
    print("\nV2 resumed training completed.")


if __name__ == "__main__":
    main()