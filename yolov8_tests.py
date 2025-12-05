from ultralytics import YOLO
import os
import shutil

model_brand = YOLO("./runs/detect/train3/weights/best.pt")  
model_year  = YOLO("./detec_year_ai/runs/detect/train/weights/best.pt") 

def get_top_class_det(res):
  
    if res.boxes is None or len(res.boxes) == 0:
        return None

    confs = res.boxes.conf.cpu().numpy()

    idx = confs.argmax()

    class_id = int(res.boxes.cls[idx].cpu().numpy())
    class_name = res.names[class_id]

    return class_name.replace(" ", "_")

def recognize_and_rename(image_path):


    res_brand = model_brand.predict(image_path)[0]
    brand_model = get_top_class_det(res_brand)
    
    if brand_model is None:
        raise ValueError("Модель не нашла автомобиль на фото (brand_model)")


    _, model_only = brand_model.split("_", 1)


    res_year = model_year.predict(image_path)[0]
    if res_year.boxes is None or len(res_year.boxes) == 0:
        raise ValueError("Модель не нашла автомобиль на фото (year_model)")

    best_class = None
    best_score = -1

    for cls_id, conf in zip(res_year.boxes.cls, res_year.boxes.conf):
        class_name = res_year.names[int(cls_id)]
        conf = float(conf)


        if class_name.startswith(model_only + "_"):
            if conf > best_score:
                best_score = conf
                best_class = class_name

    if best_class is None:

        best_class = get_top_class_det(res_year)

    # X5_2012 → 2012
    _, year = best_class.split("_", 1)


    folder = os.path.dirname(image_path)
    ext = os.path.splitext(image_path)[1]
    new_name = f"{brand_model}_{year}{ext}"
    new_path = os.path.join(folder, new_name)

    os.rename(image_path, new_path)
    return new_path


def recognize_and_rename_to_folder(image_path, output_folder="./detect_ai"):
    os.makedirs(output_folder, exist_ok=True)


    res_brand = model_brand.predict(image_path)[0]
    brand_model = get_top_class_det(res_brand)
    if brand_model is None:
        raise ValueError("Не найден автомобиль (brand_model)")
    _, model_only = brand_model.split("_", 1)


    res_year = model_year.predict(image_path)[0]
    if res_year.boxes is None or len(res_year.boxes) == 0:
        raise ValueError("Не найден автомобиль (year_model)")

    best_class = None
    best_score = -1
    for cls_id, conf in zip(res_year.boxes.cls, res_year.boxes.conf):
        class_name = res_year.names[int(cls_id)]
        conf = float(conf)
        if class_name.startswith(model_only + "_"):
            if conf > best_score:
                best_score = conf
                best_class = class_name
    if best_class is None:
        best_class = get_top_class_det(res_year)
    _, year = best_class.split("_", 1)

    ext = os.path.splitext(image_path)[1]
    new_name = f"{brand_model}_{year}{ext}"
    new_path = os.path.join(output_folder, new_name)

    shutil.copy(image_path, new_path) 
    return new_path

def process_folder(folder_path, output_folder="./detect_ai"):
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder_path):
        if file.lower().endswith(supported_ext):
            full_path = os.path.join(folder_path, file)

            try:
                new_path = recognize_and_rename_to_folder(full_path, output_folder)
                print(f"Сохранено: {file} → {new_path}")

            except Exception as e:
                print(f"Ошибка с файлом {file}: {e}")

process_folder("./runs/detect/predict5")
