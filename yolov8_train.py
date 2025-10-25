from ultralytics import YOLO
import torch

torch.cuda.empty_cache() if torch.cuda.is_available() else None
def main():
    # Загружаем базовую модель
    model = YOLO('yolov8n.pt')
    
    # Запуск обучения
    model.train(
        data='./dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,           # Максимально возможный для вашей памяти
        workers=6,          # 2-4 × количество CPU ядер
        device=0,
        lr0=0.01,
        cache='disk', 
    )

if __name__ == '__main__':
    main()