import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from vgg import vgg16
from dataset import CustomDataset
import os
import time
import numpy as np
from torch.utils.data import random_split
import argparse
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端
import matplotlib.pyplot as plt

def train(epochs=10, batch_size=4, learning_rate=0.001, valid_split=0.1, save_dir='./models'):
    # 确保模型保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保为灰度图
        transforms.Resize(256),
        transforms.RandomCrop(224),  # 随机裁剪用于数据增强
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 验证集使用的变换
    valid_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保为灰度图
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("正在加载训练数据...")
    full_trainset = CustomDataset(
        data_dir='./data/train',
        transform=transform
    )
    
    # 划分训练集和验证集
    valid_size = int(len(full_trainset) * valid_split)
    train_size = len(full_trainset) - valid_size
    
    trainset, validset = random_split(full_trainset, [train_size, valid_size])
    
    # 为验证集应用验证变换
    validset_with_transform = CustomDataset(
        data_dir='./data/train',
        transform=valid_transform
    )
    # 保持相同的索引划分
    validset = torch.utils.data.Subset(validset_with_transform, validset.indices)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    validloader = torch.utils.data.DataLoader(
        validset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    print(f"训练集样本数: {train_size}")
    print(f"验证集样本数: {valid_size}")

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载预训练模型，并设置类别数为8
    net = vgg16(pretrained=True, num_classes=8)
    net.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # 学习率调度

    # 跟踪训练过程
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    best_valid_acc = 0.0

    # 训练模型
    print("开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 9:  # 每10个bach打印一次
                print(f'[{epoch + 1}, {i + 1}] 损失: {running_loss / 10:.3f}, 准确率: {100 * correct / total:.2f}%')
                running_loss = 0.0

        train_acc = 100 * correct / total
        train_accs.append(100 * correct / total)
        train_losses.append(running_loss / len(trainloader))
        
        # 验证阶段
        net.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in validloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_time = time.time() - epoch_start
        valid_acc = 100 * correct / total
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss / len(validloader))
        
        print(f'轮次 {epoch + 1}/{epochs} 完成 | 耗时: {epoch_time:.2f}s | 训练准确率: {train_acc:.2f}% | 验证准确率: {valid_acc:.2f}%')
        
        # 保存最佳模型
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            model_path = os.path.join(save_dir, 'vgg16_best.pth')
            torch.save(net.state_dict(), model_path)
            print(f"保存最佳模型，验证准确率: {valid_acc:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'vgg16_final.pth')
    torch.save(net.state_dict(), final_model_path)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f'训练完成！总耗时: {total_time/60:.2f}分钟')
    print(f'最佳验证准确率: {best_valid_acc:.2f}%')
    print(f'最终模型保存至: {final_model_path}')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(valid_losses, label='validation loss')
    plt.title('validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='training accuracy')
    plt.plot(valid_accs, label='validation accuracy')
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='VGG16 训练脚本')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--valid-split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--save-dir', type=str, default='./models', help='模型保存目录')
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.lr,
        valid_split=args.valid_split,
        save_dir=args.save_dir
    )
