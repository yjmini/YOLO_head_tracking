from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

results = model.track(
    source="./videos/test2.mp4", 
    save=True,
    show=True,
    conf=0.5,
    tracker="bytetrack.yaml"
)
