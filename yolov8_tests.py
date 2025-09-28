from ultralytics import YOLO
import glob
model = YOLO('./runs/detect/train3/weights/best.pt')
test_folder = 'test_images'

images = glob.glob(f'{test_folder}/*.jpg') + glob.glob(f'{test_folder}/*.png')

for img in images:
    results = model(img, conf=0.1, save=True)