import os
import csv

DATA_DIR = "data/raw_images"
OUT_CSV = "data/labels.csv"

rows = []

for class_name in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    for img in os.listdir(class_path):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append([
                f"{class_name}/{img}",
                class_name
            ])

# write csv
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    writer.writerows(rows)

print(f"Generated {len(rows)} rows into {OUT_CSV}")
