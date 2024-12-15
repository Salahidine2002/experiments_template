import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)  # Max pool after conv2
        x = x.view(-1, 64 * 16 * 16)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to setup the experiment
def setup_experiment_handler(config_file_path):
    experiments = []
    with open(config_file_path, 'r') as f:
        for line in f.readlines():
            experiments.append(json.loads(line.strip()))

    if len(experiments) == 0:
        raise ValueError("The input JSON lines file must contain at least one experiment.")

    return experiments

# Generate a unique experiment directory
def generate_unique_experiment_dir(experiment_name):
    experiment_dir = os.path.join("experiments", experiment_name)
    if os.path.exists(experiment_dir):
        version = 1
        while os.path.exists(f"{experiment_dir}_v{version}"):
            version += 1
        experiment_dir = f"{experiment_dir}_v{version}"
        print(f"Experiment folder already exists! Renaming to {experiment_dir} to avoid overwriting.")
    print(f"Running {experiment_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def setup_logger(log_file_path):
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Setup directories for logs, checkpoints, and experiment config
def setup_experiment_dirs(experiment_config):
    experiment_name = experiment_config.get('experiment_name', 'default_experiment')
    experiment_dir = generate_unique_experiment_dir(experiment_name)

    log_file_name = experiment_config.get('log_file_name', 'log.txt')
    log_file_path = os.path.join(experiment_dir, log_file_name)
    with open(log_file_path, 'w') as f:
        pass
    logger = setup_logger(log_file_path)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    config_copy_path = os.path.join(experiment_dir, "config.json")
    with open(config_copy_path, 'w') as f:
        json.dump(experiment_config, f, indent=4)

    logger.info(f"Experiment setup completed for {experiment_name}. Logs saved to {log_file_path}")

    return experiment_dir, log_file_path, checkpoints_dir, logger

# Log training progress
def log_training_progress(epoch, batch, loss, train_acc, val_acc, logger):
    logger.info(f"Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}, "
                 f"Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

# Save checkpoint after every epoch
def save_checkpoint(model, optimizer, epoch, loss, checkpoints_dir, logger, checkpoint_prefix="checkpoint"):
    checkpoint_path = os.path.join(checkpoints_dir, f"{checkpoint_prefix}_epoch_{epoch}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")

# Save the best model based on validation loss
def save_best_model(model, optimizer, epoch, loss, best_loss, best_model_path, logger):
    if loss < best_loss:
        best_loss = loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, best_model_path)
        logger.info(f"Saved best model at epoch {epoch} with loss {loss:.4f} to {best_model_path}")

    return best_loss

# Train the model
def train_model(model, optimizer, epochs, train_loader, val_loader, experiment_config, logger):
    best_loss = float('inf')
    experiment_name = experiment_config['experiment_name']
    checkpoints_dir = os.path.join('experiments', experiment_name, 'checkpoints')
    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            if batch_idx % 100 == 0:  # Log every 100 batches
                train_acc = 100. * correct_train / total_train
                val_acc = validate_model(model, val_loader)
                log_training_progress(epoch, batch_idx, running_loss / (batch_idx + 1), train_acc, val_acc, logger)

        # Save checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, epoch, running_loss / len(train_loader), checkpoints_dir, logger)

        # Save the best model based on validation loss
        best_loss = save_best_model(model, optimizer, epoch, running_loss / len(train_loader), best_loss, best_model_path, logger)

    return model

# Validate the model
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100. * correct / total
    return accuracy

# Set up data loaders for CIFAR-10
def get_cifar10_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# Main entry point
if __name__ == "__main__":
    config_file = 'config.jsonl'
    experiments = setup_experiment_handler(config_file)

    for experiment_config in experiments:
        # Setup experiment directories
        experiment_dir, log_file, checkpoints_dir, logger = setup_experiment_dirs(experiment_config)

        # Set up the model, optimizer, and data loaders
        model = SimpleCNN()
        optimizer = optim.Adam(model.parameters(), lr=experiment_config['learning_rate'])

        train_loader, val_loader = get_cifar10_loaders(experiment_config['batch_size'])

        # Train the model
        trained_model = train_model(model, optimizer, experiment_config['epochs'], train_loader, val_loader, experiment_config, logger)
