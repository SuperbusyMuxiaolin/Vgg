## 直接在pytorch界面选择版本安装 [https://pytorch.org/get-started/previous-versions/]

## 或者手动安装
- 1、创建conda环境并进入
```bash
conda create --name vgg python=3.7
conda activate vgg
```

- 2、安装cudatoolkit.
```bash
conda install -c nvidia cudatoolkit=11.0
```

- 3、安装对应的cudnn.
```bash
conda install -c nvidia cudnn=8.0.4
```

- 4、安装pytorch. [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/),`+cu110`指与cuda版本11.0适配
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)
```
## 验证环境

运行以下命令进入 Python 环境：
```bash
python
```

在python环境中运行如下代码
```python
import torch
print(torch.__**version__**)
print(torch.cuda.is_available())
```

-`torch.__version__`: 输出 PyTorch 的版本号，表示 PyTorch 已正确安装。

-`torch.cuda.is_available()`: 如果安装了 GPU 版本的 PyTorch，这个命令会返回 `True`，表示可以使用 GPU；否则返回 `False`。

## 安装其他依赖项
```bash
pip install numpy pillow matplotlib
pip install torchvision scikit-learn
```
若遇到Qt相关错误，需要
```bash
sudo apt-get update
sudo apt-get install libxcb-xinerama0 libxcb-xinput0
```

## 数据预处理
将不同类别图片按照不同子文件夹形式放到`./data_original`
运行预处理脚本
```bash
python data_preprocess.py

```
## 训练
```bash
conda activate vgg
cd Vgg # 代码根目录
python train.py --epochs 50 --batch-size 16 --lr 0.0005 --valid-split 0.15 --save-dir ./models 
```

## 测试
测试单个阵型图片
```bash
python predict_single.py --image ./data/test/circle1.jpg --model ./models/vgg16_final.pth
```

测试整个数据集分类准确率
```bash
Python test.py --model ./models/vgg16_final.pth -- batchsize 16 --output ./results
```

