from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

# Train the model
model.train(
    data="/home/ssafy/work/head_tracking_260224/archive/PartB/data.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    batch=16,   # 
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

