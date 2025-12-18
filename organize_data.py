# organize_data.py
import os
import random
import shutil

print("📁 ORGANIZING DATASET")
print("=" * 50)

# المسارات
source_dir = "dataset"
target_dir = "organized_data"

# تأكد من وجود البيانات الأصلية
if not os.path.exists(source_dir):
    print("❌ ERROR: 'dataset' folder not found!")
    exit()

# إنشاء الهيكل المنظم
splits = ["train", "val", "test"]
classes = ["Acne", "Hyperpigmentation", "Nail Psoriasis", "Vitiligo"]

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

print("✅ Created folder structure")

# نسخ وتقسيم كل فئة
for cls in classes:
    print(f"\n📊 Processing: {cls}")
    
    # البحث عن المجلد المناسب (مراعاة حالة الأحرف)
    source_cls_dir = None
    for item in os.listdir(source_dir):
        if item.lower() == cls.lower().replace(" ", "_") or item.lower() == cls.lower():
            source_cls_dir = os.path.join(source_dir, item)
            break
    
    if not source_cls_dir or not os.path.exists(source_cls_dir):
        print(f"   ⚠️ Skipping: Folder not found")
        continue
    
    # جمع الصور
    images = []
    for img in os.listdir(source_cls_dir):
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images.append(img)
    
    if not images:
        print(f"   ⚠️ No images found")
        continue
    
    print(f"   📸 Found: {len(images)} images")
    
    # خلط عشوائي
    random.shuffle(images)
    
    # التقسيم: 70% تدريب، 15% تحقق، 15% اختبار
    total = len(images)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)
    
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    # دالة النسخ
    def copy_to_split(image_list, split_name):
        for img in image_list:
            src = os.path.join(source_cls_dir, img)
            dst = os.path.join(target_dir, split_name, cls, img)
            shutil.copy2(src, dst)
    
    # النسخ
    copy_to_split(train_images, "train")
    copy_to_split(val_images, "val")
    copy_to_split(test_images, "test")
    
    print(f"   ├── Train: {len(train_images)}")
    print(f"   ├── Val: {len(val_images)}")
    print(f"   └── Test: {len(test_images)}")

# عرض الإحصائيات النهائية
print("\n" + "=" * 50)
print("📈 FINAL STATISTICS")
print("=" * 50)

for split in splits:
    split_total = 0
    print(f"\n{split.upper()}:")
    
    for cls in classes:
        cls_path = os.path.join(target_dir, split, cls)
        if os.path.exists(cls_path):
            count = len([f for f in os.listdir(cls_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
            print(f"   {cls}: {count} images")
            split_total += count
    
    print(f"   Total: {split_total} images")

print("\n✅ Dataset organized successfully!")
print(f"📁 Location: {target_dir}")