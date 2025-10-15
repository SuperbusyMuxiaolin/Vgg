# 环境配置
## conda创建环境
conda create -n v python=3.9  
conda activate v
## 安装torch
注意：windows的pytorch只支持python>=3.9
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
## 安装其他依赖
pip install scikit-learn pillow matplotlib
## 检查torch环境
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

-`torch.cuda.is_available()`: 如果安装了 GPU 版本的 PyTorch，这个命令会返回 `True`，表示可以使用 GPU；否则返回 `False`

# 数据预处理
然后运行预处理脚本
```bash
python data_preprocess.py

```

# 训练
```bash
conda activate vgg
cd Vgg # 代码根目录
python train.py --epochs 20 --batch-size 16 --lr 0.0005 --valid-split 0.15 --save-dir ./models 
```

# 测试
测试单个阵型图片
```bash
python predict_single.py --image ./data/test/circle/circle_1.png --model ./models/vgg16_best.pth
```

测试整个数据集分类准确率 
```bash
python test.py --model ./models/vgg16_best.pth --batch_size 16 --output ./results
```

## 其他
# 生成sbom
C:/Users/lpl/miniforge3/Scripts/conda.exe run -p C:\Users\lpl\miniforge3 --no-capture-output python tools/generate_sbom_from_conda_spec.py -i requirements.txt -o sbom