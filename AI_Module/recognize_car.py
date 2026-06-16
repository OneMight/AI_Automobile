import time
from ultralytics import YOLO
from PIL import Image
import io

try:
    model_brand = YOLO("./runs/detect/train5/weights/best.pt") 
    model_year  = YOLO("./detec_year_ai/runs/detect/train4/weights/best.pt")
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")

def get_prediction_data(image_bytes: bytes):
    start_time = time.time()
    image = Image.open(io.BytesIO(image_bytes))
    res_brand = model_brand.predict(image, conf=0.85)[0] 
    if not res_brand.boxes or len(res_brand.boxes) == 0:
        return None
    boxes = res_brand.boxes
    sorted_indices = boxes.conf.argsort(descending=True).cpu().numpy()
    top_idx = sorted_indices[0]
    top_cls_id = int(boxes.cls[top_idx].item())
    top_conf = float(boxes.conf[top_idx].item())
    top_name = res_brand.names[top_cls_id] 
    try:
        mark, model = top_name.split("_", 1)
    except ValueError:
        mark, model = top_name, "Unknown"
    similar_models = []
    for idx in sorted_indices[1:3]: 
        cls_id = int(boxes.cls[idx].item())
        conf = float(boxes.conf[idx].item())
        name = res_brand.names[cls_id] 
        try:
            s_mark, s_model = name.split("_", 1)
        except ValueError:
            s_mark, s_model = name, "Unknown"
        similar_models.append({
            "mark": s_mark,
            "model": s_model,
            "confidence": round(conf, 4)
        })
    res_year = model_year.predict(image, conf=0.1)[0]
    best_year_str = "Unknown" 
    if res_year.boxes is not None and len(res_year.boxes) > 0:
        best_score = -1
        model_name_only = model.split("_")[-1] 
        for cls_id, conf in zip(res_year.boxes.cls, res_year.boxes.conf):
            class_name = res_year.names[int(cls_id)]
            conf = float(conf)
            if model_name_only.lower() in class_name.lower() or \
               (model_name_only.lower() == "laguna" and "lahuna" in class_name.lower()):  
                if conf > best_score:
                    best_score = conf
                    try:
                        best_year_str = class_name.split("_", 1)[1]
                    except:
                        best_year_str = class_name
        if best_year_str == "Unknown":
           return None
    end_time = time.time()
    return {
        "mark": mark,
        "model": model,
        "manufactureYear": best_year_str,
        "determinedTime": round(end_time - start_time, 2),
        "confidence": round(top_conf, 2),
        "similarModels": similar_models
    }