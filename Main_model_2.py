# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 修改为二分类问题
num_classes = 2

# 定义五层全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)
        return out

# 定义包含3个卷积模块的CNN网络
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def evaluate_performance(model, data_loader):
    model.eval()
    total, correct, auc_score = 0, 0, 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # AUC评估需要概率和真实标签
            all_labels.extend(labels.cpu().numpy())
            probs = nn.functional.softmax(outputs, dim=1)
            all_predictions.extend(probs[:, 1].cpu().numpy())

    accuracy = correct / total
    tpr = sum((predicted == 1) & (labels == 1)).item() / sum(labels == 1).item()
    tnr = sum((predicted == 0) & (labels == 0)).item() / sum(labels == 0).item()
    auc_score = roc_auc_score(all_labels, all_predictions)

    return accuracy, tpr, tnr, auc_score


model_choice = 'cnn'

# 生成随机数据和标签
# num_samples = 1000
num_features = 6

# features = torch.randn(num_samples, num_features)
# labels = torch.randint(0, num_classes, (num_samples,))

file_path = 'data/data_ml_2/data_ml_2_combination/datasets.kpl'
# file_path = './data/data_ml_1/datasets.kpl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

features = data['data']
features = torch.tensor(features)
labels = data['target']
print(sum(labels))
num_samples = len(labels)
print(num_samples)
labels = torch.tensor(labels)
# 如果选择的是CNN，调整特征的维度
if model_choice == 'cnn':
    features = features.unsqueeze(1)
# 实例化模型并移动到设备
if model_choice == 'simple_nn':
    model = SimpleNN(input_size=num_features, hidden_size=50, num_classes=num_classes).to(device)
elif model_choice == 'cnn':
    model = CNN(num_classes=num_classes).to(device)

# 移动数据到设备
features = features.to(device)
labels = labels.to(device)

# 划分数据集
train_size = int(0.5 * num_samples)
test_size = num_samples - train_size
train_dataset, test_dataset = random_split(TensorDataset(features, labels), [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练原始模型
num_epochs = 2
for epoch in range(num_epochs):
    print("这是第 {} 个epoch".format(epoch + 1))
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # 在训练集上评估性能
    train_accuracy, train_tpr, train_tnr, train_auc = evaluate_performance(model, train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}, Train TPR: {train_tpr:.4f}, '
          f'Train TNR: {train_tnr:.4f}, Train AUC: {train_auc:.4f}')

# 保存模型（除了最后一层）
state_dict = model.state_dict()
if model_choice == 'simple_nn':
    state_dict.pop('fc5.weight')
    state_dict.pop('fc5.bias')
elif model_choice == 'cnn':
    state_dict.pop('fc.weight')
    state_dict.pop('fc.bias')
torch.save(state_dict, 'data/model_saved/'+model_choice+'_model_without_last_layer.pth')

# 加载保存的模型
model.load_state_dict(torch.load('data/model_saved/'+model_choice+'_model_without_last_layer.pth'), strict=False)

# 为模型的最后一层添加新的参数
if model_choice == 'simple_nn':
    model.fc5 = nn.Linear(50, num_classes)
elif model_choice == 'cnn':
    model.fc = nn.Linear(64, num_classes)
model.to(device)

finetune_option = 3  # 1=仅微调最后一层，2=微调所有层，3=微调除了最后一层的所有层

# 根据所选微调选项设置requires_grad
for name, param in model.named_parameters():
    if finetune_option == 1:
        param.requires_grad = 'fc5' in name or 'fc' in name
    elif finetune_option == 2:
        param.requires_grad = True
    elif finetune_option == 3:
        param.requires_grad = not ('fc5' in name or 'fc' in name)

# 创建优化器
trainable_params = [param for param in model.parameters() if param.requires_grad]
if len(trainable_params) > 0:
    optimizer = optim.Adam(trainable_params, lr=0.001)
else:
    print("No parameters to optimize.")
    raise ValueError("No parameters to optimize.")

# 定义性能评估函数


# 微调模型
num_epochs_finetune = 5
for epoch in range(num_epochs_finetune):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 性能评估
    accuracy, tpr, tnr, auc = evaluate_performance(model, test_loader)
    print(f'Finetune Epoch [{epoch+1}/{num_epochs_finetune}], Loss: {loss.item():.4f}, '
          f'Accuracy: {accuracy:.4f}, TPR: {tpr:.4f}, TNR: {tnr:.4f}, AUC: {auc:.4f}')
