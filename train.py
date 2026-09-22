import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model():
    # 1. Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data processing and loading
    data_dir = "data"
    
    # Preserve full MRI brain context (avoid aggressive random cropping which removes tumors)
    data_transforms = {
        'Training': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Testing': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['Training', 'Testing']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True if x == 'Training' else False, num_workers=0)
                   for x in ['Training', 'Testing']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Training', 'Testing']}
    class_names = image_datasets['Training'].classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # 3. Model Architecture (ResNet18 with pretrained ImageNet features)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # 4. Optimizer and LR Scheduler for High Accuracy Fine-Tuning
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Use AdamW with fine-tuning learning rate (1e-4) to preserve pretrained representation while adapting
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    num_epochs = 12
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_acc = 0.0
    best_model_path = 'best_model.pth'

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Training', 'Testing']:
            if phase == 'Training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'Training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'Testing' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model with accuracy: {best_acc:.4f}")

        print()

    print(f'Training complete. Best Testing Accuracy: {best_acc:4f}')

if __name__ == '__main__':
    train_model()
