import torch
import torchvision.transforms as transforms
from vgg import vgg16
from dataset import CustomDataset
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

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
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每个类别的准确率统计
            for i, label in enumerate(labels):
                class_total[label] += 1
                class_correct[label] += (predicted[i] == label).item()

    # 输出结果
    accuracy = 100 * correct / total
    print(f'总体准确率: {accuracy:.2f}%')
    
    # 打印每个类别的准确率
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            print(f'类别 {name}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n分类报告:")
    print(report)
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='VGG16 测试脚本')
    parser.add_argument('--model', type=str, default='./models/vgg16_best.pth', help='模型路径')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--output', type=str, default='./results', help='结果输出目录')
    
    args = parser.parse_args()
    test(model_path=args.model, batch_size=args.batch_size, output_dir=args.output)
