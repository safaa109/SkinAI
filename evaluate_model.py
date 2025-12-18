import os
import torch
from torchvision import datasets, transforms, models

# مسارات البيانات
data_dir = "data"

# التحويلات للصور زي ما في validation
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]),
}

# تعريف الجهاز
device = torch.device("cpu")  # أو "cuda" لو عايزة تجربّي على GPU

# تحميل test set
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                    data_transforms['val'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# تعريف الموديل وتحميله
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = 4  # حسب الحالات عندك
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('best_skin_model.pth'))
model = model.to(device)
model.eval()

# التقييم
running_corrects = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

test_acc = running_corrects.double() / len(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")
