import torch
import torch.nn as nn
import torchvision.models as models
import os
from pathlib import Path
import numpy as np
import torch.utils.model_zoo as model_zoo

# 定义VGG网络结构
class VGG(nn.Module):
    def __init__(self, features, num_classes=8, init_weights=True, dropout_rate=0.5):
        super(VGG, self).__init__()
        self.features = features  # 特征提取部分
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 添加自适应池化，使输入尺寸更灵活
        self.classifier = nn.Sequential(  # 分类器部分
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()  # 初始化权重

    def forward(self, x):
        x = self.features(x)  # 提取特征
        x = self.avgpool(x)  # 添加自适应池化
        x = torch.flatten(x, 1)  # 展平
        x = self.classifier(x)  # 分类
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 根据配置生成层
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1  # 修改输入通道数为1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 预训练模型的URL
model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

# 配置列表
cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    

def vgg16(pretrained=False, **kwargs):
    """VGG 16层模型

    Args:
        pretrained (bool): 如果为True，返回在ImageNet上预训练的模型
        batch_norm (bool): 如果为True，使用批归一化
    """
    batch_norm = kwargs.pop('batch_norm', False) # 从kwargs中提取batch_norm参数
    
    if batch_norm:
        model = VGG(make_layers(cfgs, batch_norm=True), **kwargs)
        model_name = 'vgg16_bn'
    else:
        model = VGG(make_layers(cfgs), **kwargs)
        model_name = 'vgg16'
        
    if pretrained:
        _load_pretrained(model, model_name, **kwargs)
    return model

def _load_pretrained(model, model_name, **kwargs):
    """加载预训练权重并进行适当修改"""
    try:
        # 预训练权重保存目录
        pretrained_dir = Path('./pretrained')
        pretrained_dir.mkdir(exist_ok=True)
        
        # 构建本地权重文件路径
        local_weight_path = pretrained_dir / f"{model_name}.pth"
        
        # 检查本地是否已有权重文件
        if local_weight_path.exists():
            print(f"从本地加载预训练权重: {local_weight_path}")
            state_dict = torch.load(local_weight_path, map_location='cpu') # 返回字典，包含模型各层的tensor
        else:
            print(f"本地未找到预训练权重，从网络下载...")
            url = model_urls.get(model_name, None)
            if url is None:
                print(f"无法找到{model_name}的预训练权重URL，将随机初始化模型")
                return
            try:
                state_dict = model_zoo.load_url(url, progress=True)
                # 保存到本地
                torch.save(state_dict, local_weight_path)
                print(f"预训练权重已下载并保存至: {local_weight_path}")
            except Exception as e:
                print(f"下载预训练权重失败: {e}")
                print("将随机初始化模型")
                return
        
        # 修改第一层卷积层权重，从3通道变为1通道
        # 方法是对RGB三通道权重取平均
        if 'features.0.weight' in state_dict:
            conv1_weight = state_dict['features.0.weight']
            state_dict['features.0.weight'] = conv1_weight.mean(dim=1, keepdim=True)
            print(f"已将第一层卷积从3通道调整为1通道")
        
        # 检查最后一层分类器是否需要调整
        num_classes = kwargs.get('num_classes')  # 默认为8类
        if 'classifier.6.weight' in state_dict and state_dict['classifier.6.weight'].size(0) != num_classes:
            # 获取原始分类器权重和偏置的形状
            old_num_classes = state_dict['classifier.6.weight'].size(0)
            fc_features = state_dict['classifier.6.weight'].size(1)
            
            print(f"调整最后一层从{old_num_classes}类到{num_classes}类")
            
            # 如果新的类别小于原始类别，直接截取
            if num_classes <= old_num_classes:
                state_dict['classifier.6.weight'] = state_dict['classifier.6.weight'][:num_classes]
                state_dict['classifier.6.bias'] = state_dict['classifier.6.bias'][:num_classes]
            else:
                # 否则需要创建新的权重，部分保留原始权重
                new_weight = torch.zeros(num_classes, fc_features)
                new_bias = torch.zeros(num_classes)
                
                # 复制原始权重到新权重的前几个类别
                new_weight[:old_num_classes] = state_dict['classifier.6.weight']
                new_bias[:old_num_classes] = state_dict['classifier.6.bias']
                
                # 随机初始化剩余部分
                if old_num_classes < num_classes:
                    nn.init.normal_(new_weight[old_num_classes:], 0, 0.01)
                
                state_dict['classifier.6.weight'] = new_weight
                state_dict['classifier.6.bias'] = new_bias
        
        # 加载修改后的权重
        try:
            # 使用strict=False允许模型结构与权重不完全匹配(例如添加了avgpool层)
            model.load_state_dict(state_dict, strict=False)
            print("预训练权重加载成功")
        except Exception as e:
            print(f"加载预训练权重时出现错误: {e}")
            print("模型将部分使用随机初始化的权重")
        
    except ImportError:
        print("警告: 缺少必要的库，无法加载预训练模型")
    except Exception as e:
        print(f"加载预训练模型时出错: {e}")
