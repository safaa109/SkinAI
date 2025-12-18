import splitfolders

input_folder = "dataset"  # فولدر الصور بتاعتك
output_folder = "data"    # الفولدر الجديد اللي هيبقى فيه train/val/test

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.2, 0.1))
print("Done! Data is split into train, val, test folders.")
