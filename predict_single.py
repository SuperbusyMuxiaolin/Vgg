import os
import torch
import argparse
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from vgg import vgg16

def predict_formation(image_path, model_path='./models/vgg16_best.pth'):
    """
    预测单个图片的阵型
    
    参数:
        image_path: 图片路径
        model_path: 模型路径
    """
    # 类别名称
    class_names = ['circle', 'column', 'cross', 'line', 'square', 'staggered', 'triangle', 'vformation']
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片文件 {image_path} 不存在!")
        return
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    try:
        # 创建模型实例
        net = vgg16(pretrained=False, num_classes=len(class_names))
        # 加载预训练权重
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 设为评估模式
    net.to(device)
    net.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保为灰度图
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 加载并预处理图像
    try:
        image = Image.open(image_path).convert('RGB')
        # 转换图像
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"图像处理失败: {e}")
        return
    
    # 预测
    with torch.no_grad():
        outputs = net(image_tensor)
        _, predicted = torch.max(outputs, 1)
        
        # 获取预测类别和概率
        predicted_idx = predicted.item()
        predicted_class = class_names[predicted_idx]
        
        # 获取所有类别的概率
        probabilities = nn.functional.softmax(outputs, dim=1)[0]
        probs = {class_names[i]: float(probabilities[i]) * 100 for i in range(len(class_names))}
        
        # 排序并显示前3个预测
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        print("\n预测结果:")
        print(f"图像 '{os.path.basename(image_path)}' 的阵型是: {predicted_class}")
        print("\n前三预测及概率:")
        for i, (cls, prob) in enumerate(sorted_probs[:3]):
            print(f"{i+1}. {cls}: {prob:.2f}%")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预测单个图片的阵型')
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--model', type=str, default='./models/vgg16_best.pth', help='模型路径')
    
    args = parser.parse_args()
    print(f"预测图像: {args.image}")
    print(f"使用模型: {args.model}")
    predict_formation(args.image, args.model)