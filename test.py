"""
测试器模块
"""
import os
import unet
import torch
import dataset
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 测试器
class Tester:
    def __init__(self, path, model,img_save_path):
        """

        path: 数据集路径
        model: 模型路径
        img_save_path: 运行结果保存路径

        """
        self.path = path
        self.model = model
        self.img_save_path = img_save_path
        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化网络模型
        self.net = unet.UNet().to(self.device)
        # 读取训练好的参数
        self.net.load_state_dict(torch.load(self.model))
        # 损失函数
        self.loss_func = nn.BCELoss()
        # 加载测试集
        self.loader = DataLoader(dataset.Datasets(path), batch_size=1, shuffle=True, num_workers=1)

    def test(self):
        epoch = 1
        for inputs, labels in tqdm(self.loader,ascii=True, total=len(self.loader)):
            # 将图像于标签存储到CPU/GPU上
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = self.net(inputs)
            # 输入的图像，取第一张
            x = inputs[0]
            # 生成的图像，取第一张
            x_ = out[0]
            # 标签的图像，取第一张
            y = labels[0]
            # 三张图，从第0轴拼接起来，再保存
            img = torch.stack([x, x_, y], 0)
            save_image(img.cpu(), os.path.join(self.img_save_path, f"{epoch}.png"))
            epoch = epoch + 1



if __name__ == '__main__':
    t = Tester(r"C:\Users\dell\Desktop\pythonprojects\Pytorch-UNet-Retina-master\DRIVE\test",r'./model_300_0.07029879093170166.pth',img_save_path='./test_img')
    t.test()