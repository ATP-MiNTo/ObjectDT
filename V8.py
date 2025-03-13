from config import API_KEY,Space_KEY,Project_KEY,Version_KEY
from roboflow import Roboflow
import subprocess, torch, ultralytics

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(Space_KEY).project(Project_KEY)
version = project.version(Version_KEY)
dataset = version.download("yolov11")

# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"Number of GPUs: {torch.cuda.device_count()}")
# print(f"Current GPU device: {torch.cuda.current_device()}")

subprocess.run([
    "yolo", "task=detect", "mode=train",
    "model=yolo11n.pt", f"data={dataset.location}/data.yaml", 
    "epochs=40", "imgsz=640", "plots=True", "device=0", "batch=50",
    "conf=0.4", "iou=0.45","pretrained=False"
])
