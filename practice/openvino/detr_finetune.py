import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
from torch.utils.tensorboard import SummaryWriter

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_urls, annotations, transform=None):
        self.image_urls = image_urls
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        image = Image.open(requests.get(self.image_urls[idx], stream=True).raw)
        if self.transform:
            image = self.transform(image)
        target = self.annotations[idx]
        return image, target

# Example data (replace with your own data)
image_urls = ['https://example.com/image1.jpg', 'https://example.com/image2.jpg']
annotations = [
    {'boxes': [[10, 20, 30, 40]], 'labels': },
    {'boxes': [[15, 25, 35, 45]], 'labels': }
]

# Transformations
transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor()
])

# Create dataset and dataloader
dataset = CustomDataset(image_urls, annotations, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load pre-trained model
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

# TensorBoard writer
writer = SummaryWriter('runs/detr_finetuning')

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(dataloader):
        # Prepare inputs
        inputs = {'pixel_values': images, 'labels': targets}
        
        # Forward pass
        outputs = model(**inputs)
        
        # Compute loss
        loss = outputs.loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Log loss to TensorBoard
        if i % 10 == 9:  # Log every 10 batches
            writer.add_scalar('training loss', running_loss / 10, epoch * len(dataloader) + i)
            running_loss = 0.0
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

print("Fine-tuning completed!")
writer.close()
