from ultralytics import YOLO
import torch

torch.cuda.empty_cache() if torch.cuda.is_available() else None
def main():
    # model = YOLO('./runs/detect/train2/weights/best.pt')
    model = YOLO('yolov8m.pt')
    model.train(
        data='./dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,          
        workers=4,          
        device=0,
        lr0=0.01,
        cache='disk', 
    )

if __name__ == '__main__':
    main()