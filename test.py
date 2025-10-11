import torch
import torchvision.transforms as transforms
from vgg import vgg16
from dataset import CustomDataset
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
from sklearn.metrics import confusion_matrix, classification_report
import random
from PIL import Image
import argparse

def test(model_path='./models/vgg16_best.pth', batch_size=4, output_dir='./results'):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载测试数据
    testset = CustomDataset(data_dir='./data/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    print(f"测试集样本数: {len(testset)}")
    class_names = testset.classes
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    net = vgg16(pretrained=False, num_classes=len(class_names))
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载模型: {model_path}")
    else:
        print(f"错误: 模型文件不存在: {model_path}")
        return
        
    net.to(device)
    net.eval()

    # 测试模型
    all_preds, all_labels = [], []
    correct, total = 0, 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    # 存储错误预测的样本信息
    misclassified_samples = []
    image_paths = []
    
    # 收集所有测试图像的路径
    for idx in range(len(testset)):
        image_path = testset.images[idx]
        image_paths.append(image_path)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 记录错误预测的样本
            for j in range(images.size(0)):
                if predicted[j] != labels[j]:
                    batch_idx = i * batch_size + j
                    if batch_idx < len(image_paths):
                        misclassified_samples.append({
                            'image_path': image_paths[batch_idx],
                            'true_label': class_names[labels[j].item()],
                            'pred_label': class_names[predicted[j].item()],
                            'true_idx': labels[j].item(),
                            'pred_idx': predicted[j].item()
                        })
            
            # 每个类别的准确率统计
            for j, label in enumerate(labels):
                class_total[label] += 1
                class_correct[label] += (predicted[j] == label).item()

    # 输出结果
    accuracy = 100 * correct / total
    print(f'总体准确率: {accuracy:.2f}%')
    
    # 打印每个类别的准确率
    class_accuracies = []
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            class_accuracies.append(acc)
            print(f'类别 {name}: {acc:.2f}%')
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n分类报告:") 
    print(report)
    
    # 1. 可视化: 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在混淆矩阵中标注数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"混淆矩阵已保存到: {confusion_matrix_path}")
    plt.close()
    
    # 2. 可视化: 类别准确率
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_accuracies)
    plt.axhline(y=accuracy, color='r', linestyle='-', label=f'平均准确率: {accuracy:.2f}%')
    plt.ylabel('准确率 (%)')
    plt.title('各类别准确率')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    accuracy_chart_path = os.path.join(output_dir, 'class_accuracy.png')
    plt.savefig(accuracy_chart_path)
    print(f"类别准确率图表已保存到: {accuracy_chart_path}")
    plt.close()
    
    # 3. 可视化: 错误预测样本展示
    if misclassified_samples:
        # 随机选择最多9个错误预测的样例
        samples_to_show = min(9, len(misclassified_samples))
        random_samples = random.sample(misclassified_samples, samples_to_show)
        
        # 创建图像网格
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, sample in enumerate(random_samples):
            if i < samples_to_show:
                img = Image.open(sample['image_path']).convert('RGB')
                axes[i].imshow(img)
                axes[i].set_title(f"真: {sample['true_label']}\n预测: {sample['pred_label']}")
                axes[i].axis('off')
        
        # 隐藏未使用的子图
        for i in range(samples_to_show, 9):
            axes[i].axis('off')
        
        plt.tight_layout()
        errors_path = os.path.join(output_dir, 'misclassified_samples.png')
        plt.savefig(errors_path)
        print(f"错误预测样例已保存到: {errors_path}")
        plt.close()
    
    # 4. 可视化: 预测概率分布示例
    with torch.no_grad():
        # 随机选择一批数据
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 为每个样本创建概率分布图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i in range(min(4, len(images))):
            probs = probabilities[i].cpu().numpy()
            pred_idx = torch.argmax(probabilities[i]).item()
            true_idx = labels[i].item()
            
            # 概率条形图
            axes[i].bar(range(len(class_names)), probs)
            axes[i].set_xticks(range(len(class_names)))
            axes[i].set_xticklabels(class_names, rotation=45)
            axes[i].set_ylim([0, 1])
            
            color = 'green' if pred_idx == true_idx else 'red'
            axes[i].set_title(f"真实: {class_names[true_idx]}, 预测: {class_names[pred_idx]}", 
                             color=color)
        
        plt.tight_layout()
        probs_path = os.path.join(output_dir, 'probability_distributions.png')
        plt.savefig(probs_path)
        print(f"预测概率分布已保存到: {probs_path}")
        plt.close()
        
    # 将分类报告保存到output_dir
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w',encoding='utf-8') as f:
        f.write(f'总体准确率: {accuracy:.2f}%\n\n')
        f.write('各类别准确率:\n')
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                f.write(f'类别 {name}: {100 * class_correct[i] / class_total[i]:.2f}%\n')
        f.write('\n分类报告:\n')
        f.write(report)
    print(f"分类报告已保存到: {report_path}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGG16 测试脚本')
    parser.add_argument('--model', type=str, default='./models/vgg16_best.pth', help='模型路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--result', type=str, default='./results', help='结果保存路径')

    args = parser.parse_args()
    test(model_path=args.model, batch_size=args.batch_size, output_dir=args.result)