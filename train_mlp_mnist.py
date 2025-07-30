import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import wandb

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.model(x)

def train_mlp_mnist():
    config = {
        "epochs": 1, # ここを1に修正
        "batch_size": 32,
        "learning_rate": 0.001
    }
    wandb.init(project="physical_ai_final_challenge", config=config, mode="offline")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    model.train()
    for epoch in range(wandb.config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                wandb.log({"loss": loss.item()})

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    wandb.log({"test_accuracy": accuracy, "test_loss": test_loss})

    model_dir = './model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'mnist_mlp_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    wandb.finish()

if __name__ == '__main__':
    train_mlp_mnist()
