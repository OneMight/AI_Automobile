import os

label_dirs = ["dataset/train/labels", "dataset/valid/labels", "dataset/test/labels"]

for d in label_dirs:
    for file in os.listdir(d):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(d, file)
        with open(path) as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls > 0:  # так как должен быть только 0
                    print(f"❌ Ошибка в {path}, строка {i}: класс {cls}")