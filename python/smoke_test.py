from ultralytics import YOLO


def main():
    print("=" * 70)
    print("YOLOv8 SEGMENTATION SMOKE TEST")
    print("=" * 70)

    model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data="clean_data.yaml",
        epochs=1,
        imgsz=416,
        batch=4,
        device=0,
        workers=0,
        fraction=0.02,
        cache=False,
        amp=True,
        pretrained=True,
        project="runs/mushroom",
        name="smoke_test",
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETED")
    print("=" * 70)

    print(results)


if __name__ == "__main__":
    main()
