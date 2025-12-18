import tensorflow as tf

# مسار الداتا بعد التقسيم
data_dir = "data"

# حجم الصورة النهائي
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# تحميل البيانات من الفولدرات (train / val / test)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir + "/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir + "/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir + "/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# تحسين الأداء عبر الكاش والprefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# طباعة أسماء الكلاسات للتأكد
class_names = train_ds.class_names
print("Classes:", class_names)

print("\nDataset is ready!")
