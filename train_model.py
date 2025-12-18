import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# مسارات البيانات
data_dir = "data"

# التحويلات للصور
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]),
}

# تحميل البيانات
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
               for x in ['train', 'val']}

# تعريف النموذج (ResNet18)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = 4 # عندنا 4 أصناف: acne,hyperpigmentation,Nail_psoriasis,Vitiligo
model.fc = nn.Linear(num_ftrs, num_classes)

# نشتغل على CPU دلوقتي لتجنب مشاكل CUDA
device = torch.device("cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# عدد الـ epochs اللي عايزينه
num_epochs = 10

# متغير لتتبع أفضل دقة
best_val_acc = 0.0
best_model_path = "best_skin_model.pth"

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    for phase in ['train', 'val']:
        running_loss = 0.0
        running_corrects = 0

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # لو دقة الـ validation أحسن من قبل، نحفظ النموذج
        if phase == 'val' and epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated and saved to {best_model_path}")

print("\nTraining finished!")
print(f"Best validation accuracy: {best_val_acc:.4f}")

