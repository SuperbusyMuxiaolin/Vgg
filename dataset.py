import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
from collections import Counter

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        参数:
            data_dir: 数据集根目录，下面应该包含8个子文件夹（对应8个类别）
            transform: 数据预处理方式
        目录结构示例:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
                ...
            class2/
                img1.jpg
                img2.jpg
                ...
            ...
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 验证数据目录是否存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
            
        # 获取类别名称（只包含目录）
        self.classes = [d for d in sorted(os.listdir(data_dir)) 
                       if os.path.isdir(os.path.join(data_dir, d))]
                       
        if not self.classes:
            raise ValueError(f"在{data_dir}中没有找到任何类别目录")
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # 收集所有图像路径和对应的标签
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
                
            # 只接受图像文件
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(valid_extensions):
                    img_path = os.path.join(cls_dir, img_name)
                    # 验证文件可读
                    try:
                        with Image.open(img_path) as img:
                            pass  # 只测试能否打开
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])
                    except Exception as e:
                        print(f"警告: 无法读取图像 {img_path}: {str(e)}")
        
        if len(self.images) == 0:
            raise ValueError(f"没有找到有效的图像文件在 {data_dir}")
            
        # 打印数据集统计信息
        label_counts = Counter(self.labels)
        print(f"数据集 '{data_dir}' 加载完成:")
        print(f"- 总样本数: {len(self.images)}")
        print(f"- 类别数量: {len(self.classes)}")
        for cls_idx, count in sorted(label_counts.items()):
            cls_name = self.classes[cls_idx]
            print(f"  - {cls_name}: {count}张图像")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # 读取图像并转换为灰度图
            image = Image.open(img_path).convert('L')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"错误: 处理图像 {img_path} 时出错: {str(e)}")
            # 返回一个占位图像和标签
            if self.transform:
                placeholder = torch.zeros((1, 224, 224))
            else:
                placeholder = Image.new('L', (224, 224), 0)
            return placeholder, label
            
    def get_classes(self):
        """返回类别名称列表"""
        return self.classes
