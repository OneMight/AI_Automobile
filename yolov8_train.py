from ultralytics import YOLO

# Загружаем базовую модель (можно выбрать 'yolov8n.pt', 'yolov8s.pt' и т.д.)
model = YOLO('yolov8n.pt')

# Запуск обучения
model.train(
    data='./dataset/data.yaml',   # путь к файлу с описанием данных
    epochs=70,          # количество эпох
    imgsz=640,          # размер изображений
    batch=32,           # размер батча
    lr0=0.001            # learning rate
)