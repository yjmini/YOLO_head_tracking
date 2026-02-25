from ultralytics import YOLO

model = YOLO("./runs/detect/train2/weights/best.pt")

results = model.track(
    source="./videos/test3.mp4", 
    save=True,
    show=True,
    conf=0.7,
    tracker="bytetrack.yaml"
)
