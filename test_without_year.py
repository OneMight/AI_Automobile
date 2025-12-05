from ultralytics import YOLO

if __name__ == '__main__':
    # Загружаем модель
    model = YOLO("./detec_year_ai/runs/detect/train/weights/best.pt")

    # Валидируем на тестовом датасете
    results = model.val(data="./detec_year_ai/dataset/data.yaml")  # YAML с путями к тесту и классами

    # Метрики детекции
    print("mAP@0.5:", results.metrics[0]['map50'])
    print("mAP@0.5:0.95:", results.metrics[0]['map50_95'])
    print("Precision:", results.metrics[0]['precision'])
    print("Recall:", results.metrics[0]['recall'])
    print("F1:", results.metrics[0]['f1'])
