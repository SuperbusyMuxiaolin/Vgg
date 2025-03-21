import os
import shutil
import random
import sys
import traceback

print("DEBUG: 脚本开始执行")  # 添加这行作为第一个输出

def split_dataset(source_dir, train_dir, test_dir, test_ratio=0.2):
    try:
        print(f"开始处理数据集...")
        print(f"源数据目录: {source_dir}")
        print(f"训练集目录: {train_dir}")
        print(f"测试集目录: {test_dir}")

        if not os.path.exists(source_dir):
            print(f"错误：源数据目录 {source_dir} 不存在！")
            return False  # 修复这里的返回值，从n改为False

        # 创建训练集和测试集目录
        os.makedirs(train_dir, exist_ok=True)
        print(f"确保训练集目录存在: {train_dir}")
        os.makedirs(test_dir, exist_ok=True)
        print(f"确保测试集目录存在: {test_dir}")

        # 遍历每个类别
        class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        if not class_names:
            print("警告：源目录中没有找到任何类别子目录")
            return False
            
        print(f"找到以下类别: {class_names}")

        for class_name in class_names:
            print(f"\n处理类别: {class_name}")
            class_dir = os.path.join(source_dir, class_name)
            
            # 创建对应的训练集和测试集目录
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # 获取该类别下所有图像
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                print(f"警告：{class_name}类别中没有找到图片文件")
                continue
                
            print(f"该类别共有 {len(images)} 张图片")
            
            num_test = max(1, int(len(images) * test_ratio))  # 至少选择1张测试图片
            print(f"将选择 {num_test} 张图片作为测试集")
            
            # 随机选择测试集图像
            test_images = random.sample(images, num_test)
            
            # 复制文件到对应目录
            train_count = 0
            test_count = 0
            for img in images:
                src = os.path.join(class_dir, img)
                if img in test_images:
                    dst = os.path.join(test_class_dir, img)
                    test_count += 1
                else:
                    dst = os.path.join(train_class_dir, img)
                    train_count += 1
                shutil.copy2(src, dst)
            
            print(f"训练集: {train_count} 张")
            print(f"测试集: {test_count} 张")
        return True
    except Exception as e:
        print(f"发生错误: {str(e)}")
        traceback.print_exc()  # 打印完整的堆栈跟踪
        return False

if __name__ == '__main__':
    try:
        print("="*50)
        print("数据集划分脚本")
        print("="*50)
        
        # 立即刷新输出缓冲区
        sys.stdout.flush()
        
        print("脚本开始执行...")
        print(f"Python 版本: {sys.version}")
        
        # 设置随机种子以确保可重复性
        random.seed(42)
        print("随机种子已设置")
        
        # 获取绝对路径
        current_dir = os.path.abspath(os.getcwd())
        print(f"当前工作目录: {current_dir}")
        
        source_dir = os.path.abspath('./data_original')
        train_dir = os.path.abspath('./data/train')
        test_dir = os.path.abspath('./data/test')
        
        print(f"数据目录的绝对路径: {source_dir}")
        
        if not os.path.exists(source_dir):
            print(f"错误: 数据目录 {source_dir} 不存在!")
            sys.exit(1)
            
        print("\n数据目录内容:")
        for item in os.listdir(source_dir):
            path = os.path.join(source_dir, item)
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                print(f"- {item}/ ({count}个文件)")
            else:
                print(f"- {item}")
            
        result = split_dataset(source_dir, train_dir, test_dir, test_ratio=0.2)
        if result:
            print('\n数据集划分成功！')
        else:
            print('\n数据集划分失败！')
            
    except Exception as e:
        print("="*50)
        print(f"错误详情:")
        traceback.print_exc()
        print("="*50)
        sys.exit(1)
