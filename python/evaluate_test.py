from ultralytics import YOLO


def main():
    model = YOLO(
        r"runs\segment\runs\mushroom_training_v2\yolov8n_seg_v2\weights\best.pt"
    )

    print("=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    metrics = model.val(
        data="clean_data.yaml",
        split="test",
        imgsz=512,
        batch=4,
        device=0,
        workers=0,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETED")
    print("=" * 70)

    print("\nBOX METRICS")
    print("Precision :", metrics.box.mp)
    print("Recall    :", metrics.box.mr)
    print("mAP50     :", metrics.box.map50)
    print("mAP50-95  :", metrics.box.map)

    print("\nMASK METRICS")
    print("Precision :", metrics.seg.mp)
    print("Recall    :", metrics.seg.mr)
    print("mAP50     :", metrics.seg.map50)
    print("mAP50-95  :", metrics.seg.map)

    print("\nPER-CLASS RESULTS")
    print(metrics)


if __name__ == "__main__":
    main()
