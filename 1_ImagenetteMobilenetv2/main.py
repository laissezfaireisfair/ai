import torch
import torch.nn as nn
import torchvision as tv
import tqdm

size = (224, 224)
magic_pytorch_mean = [0.485, 0.456, 0.406]
magic_pytorch_std = [0.229, 0.224, 0.225]
batch_size = 64
learning_rate = 0.005
momentum = 0.9
epochs = 10
device_name = 'cuda'  # Might be set to 'cpu'

model = tv.models.mobilenet_v2(weights=tv.models.MobileNet_V2_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 10)
device = torch.device(device_name)
model = model.to(device)

dataset_transform = tv.transforms.Compose([
    tv.transforms.Resize(size=size),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=magic_pytorch_mean, std=magic_pytorch_std)
])

# On first run need download=True
train_dataset = tv.datasets.Imagenette(root='.', split='train', transform=dataset_transform)
test_dataset = tv.datasets.Imagenette(root='.', split='val', transform=dataset_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                           pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                          pin_memory=True)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

for _ in range(epochs):
    model.train()
    for image, label in (pbar := tqdm.tqdm(train_loader)):
        optimizer.zero_grad()
        output = model(image.to(device))
        loss = criterion(output, label.to(device))
        loss.backward()
        optimizer.step()
        pbar.set_description(f'loss: {loss.item():.3f}')
    model.eval()

torch.save(model.state_dict(), 'model_state')
