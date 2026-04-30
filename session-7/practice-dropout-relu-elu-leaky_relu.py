import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt


# 1. 定义自定义Dataset类
class MNISTCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        参数:
            csv_file: CSV文件路径
            transform: 可选的图像变换
        """
        # 读取CSV文件
        data = pd.read_csv(csv_file)

        # 第一列是标签，其余是像素值
        self.labels = data.iloc[:, 0].values.astype(np.int64)
        self.images = data.iloc[:, 1:].values.astype(np.float32)

        # 将像素值归一化到[0, 1]范围
        self.images = self.images / 255.0

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 重塑为28x28的图像
        image = image.reshape(28, 28)

        # 转换为PyTorch张量
        image = torch.from_numpy(image).float()
        label = torch.tensor(label, dtype=torch.long)

        # 添加通道维度 (1, 28, 28)
        image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, label


# Basic neural network with ReLU + Dropout
class BasicReLUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(BasicReLUNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.dropout(x)  # Apply dropout after activation
        x = self.fc2(x)
        return x

# Neural network with Leaky ReLU + Dropout
class LeakyReLUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5, negative_slope=0.01):
        super(LeakyReLUNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.negative_slope = negative_slope
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Neural network with ELU + Dropout
class ELUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5, alpha=1.0):
        super(ELUNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.alpha = alpha
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x, alpha=self.alpha)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Define a function to train and evaluate models
def train_and_evaluate(model, train_loader, test_loader, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the inputs
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.view(inputs.size(0), -1)  # Flatten the inputs
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_accuracies

# Example usage
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 定义变换 - 保持和之前相同的归一化参数
    transform = None  # 我们已经在Dataset中做了归一化处理

    # 使用本地CSV文件创建数据集
    train_dataset = MNISTCSVDataset("mnist_train.csv", transform=transform)
    test_dataset = MNISTCSVDataset("mnist_test.csv", transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define models
    input_dim = 28 * 28  # MNIST images are 28x28
    hidden_dim = 128
    output_dim = 10  # 10 digits

    relu_model = BasicReLUNet(input_dim, hidden_dim, output_dim)
    leaky_relu_model = LeakyReLUNet(input_dim, hidden_dim, output_dim)
    elu_model = ELUNet(input_dim, hidden_dim, output_dim)

    # Train and evaluate models
    print("Training with ReLU...")
    relu_losses, relu_accs = train_and_evaluate(relu_model, train_loader, test_loader)

    print("\nTraining with Leaky ReLU...")
    leaky_losses, leaky_accs = train_and_evaluate(leaky_relu_model, train_loader, test_loader)

    print("\nTraining with ELU...")
    elu_losses, elu_accs = train_and_evaluate(elu_model, train_loader, test_loader)

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(relu_losses, label='ReLU')
    plt.plot(leaky_losses, label='Leaky ReLU')
    plt.plot(elu_losses, label='ELU')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(relu_accs, label='ReLU')
    plt.plot(leaky_accs, label='Leaky ReLU')
    plt.plot(elu_accs, label='ELU')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        "./lecture-dropout-relu-elu-leaky_relu.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
