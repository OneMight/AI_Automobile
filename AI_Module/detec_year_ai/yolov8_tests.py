from ultralytics import YOLO
import glob
import os
import shutil
import cv2
import random
import math

model = YOLO('./runs/detect/train/weights/best.pt')

test_folder = '../downloaded_images/Renault/Megane'
number_of_class = 12
car_model = 'Renault_Megane'
name_of_car = f'{car_model}_Megane'
output_folder = f'../results_sorted/{name_of_car}_AutoDetect'
os.makedirs(output_folder, exist_ok=True)

nullable_folder = os.path.join(output_folder, 'nullable')
os.makedirs(nullable_folder, exist_ok=True)

dataset_folders = {}
for split in ['train', 'val', 'test']:
    dataset_folders[split] = os.path.join(output_folder, split)
    os.makedirs(dataset_folders[split], exist_ok=True)
    os.makedirs(os.path.join(dataset_folders[split], 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folders[split], 'labels'), exist_ok=True)

images = glob.glob(f'{test_folder}/**/*.jpg', recursive=True) + \
         glob.glob(f'{test_folder}/**/*.png', recursive=True)

random.shuffle(images)

total = len(images)
n_train = math.ceil(total * 0.7)
n_val = math.ceil(total * 0.2)
n_test = total - n_train - n_val

split_indices = {'train': n_train, 'val': n_val, 'test': n_train + n_val}

for idx, img_path in enumerate(images):
    if idx < split_indices['train']:
        split = 'train'
    elif idx < split_indices['val'] + split_indices['train']:
        split = 'val'
    else:
        split = 'test'

    img_name = os.path.basename(img_path)
    ext = os.path.splitext(img_name)[1]
    img_name_car = f'{name_of_car}_{idx:04d}{ext}'

    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (640, 640))

    height, width, _ = img.shape

    results = model(img, conf=0.7, save=False, verbose=False)

    if len(results[0].boxes) > 0:
        label_lines = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name_of_car, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            label_lines.append(f"{number_of_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        img_save_path = os.path.join(dataset_folders[split], 'images', img_name_car)
        cv2.imwrite(img_save_path, img)

        label_save_path = os.path.join(dataset_folders[split], 'labels', os.path.splitext(img_name_car)[0] + '.txt')
        with open(label_save_path, 'w') as f:
            f.write('\n'.join(label_lines))
    else:
        shutil.copy(img_path, os.path.join(nullable_folder, img_name))

print("\n✅ Обработка, разметка и разбиение на train/val/test завершено! Все изображения уменьшены до 640x640.")
