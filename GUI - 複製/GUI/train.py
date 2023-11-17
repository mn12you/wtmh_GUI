import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report, roc_curve, auc

train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据变换，可以根据需要进行调整
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # 调整图像大小
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert to a PyTorch tensor
])

# 加载数据集
data_root = "D:\\Desktop\\AHA CWT images\\cwt112less"
dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
# 计算训练集和测试集的划分
total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

# 划分数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建训练集和测试集的数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(p=0.25)

        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout4 = nn.Dropout(p=0.25)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=588, out_features=3 * 16)
        self.fc2 = nn.Linear(in_features=3 * 16, out_features=4)

        self.classes = ['F', 'N', 'Q','V']

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)

        x=self.flatten(x)

        
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x
    

model = ConvNet().to(device)  # Move the model to the CUDA device
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 120  # You can adjust this based on your training needs

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Calculate training accuracy
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_accuracy.append(100 * correct / total)

    # Test the model
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_losses.append(running_loss / len(test_loader))
    test_accuracy.append(100 * correct / total)

    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {train_losses[-1]:.4f} - Train Acc: {train_accuracy[-1]:.2f}% - Test Acc: {test_accuracy[-1]:.2f}%')

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
all_predicted = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy on the test dataset: {accuracy:.2f}%')

torch.save(model.state_dict(), 'model112.pth')

# Calculate the confusion matrix
confusion = confusion_matrix(all_labels, all_predicted)


# Print the confusion matrix

print("Confusion Matrix:")
print(confusion)

# ...

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accuracy, label="Train Accuracy")
plt.plot(test_accuracy, label="Test Accuracy")
plt.title("Accuracy Curves")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.show()

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Display the classification report
report = classification_report(all_labels, all_predicted, target_names=model.classes)
print("Classification Report:")
print(report)

# Plot the confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = torch.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(confusion, classes=model.classes)