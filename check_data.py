import os

base_folder = "data"  # الفولدر اللي فيه train/val/test

for split in ["train", "val", "test"]:
    print(f"\n=== {split.upper()} ===")
    split_path = os.path.join(base_folder, split)
    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            print(f"{cls}: {len(files)} images")
