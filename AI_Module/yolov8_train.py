from ultralytics import YOLO
import torch

torch.cuda.empty_cache() if torch.cuda.is_available() else None
def main():
    model = YOLO('yolov8n.pt') 
    model.train(
        data='./dataset/data.yaml',
        epochs=50,     
        imgsz=640,
        batch=16,       
        workers=4,     
        device=0,
        lr0=0.001,
        cache = 'disk',
        half=True,
       hsv_h=0.015,
        hsv_s=0.9,       
        hsv_v=0.9,    
        flipud=0.0,
        fliplr=0.5,
        shear=0.2,     
        perspective=0.001, 
        scale=0.7
             
    )
if __name__ == '__main__':
    main()
