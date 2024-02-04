import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class Drop(nn.Module):
    # drop_rate : 1-keep_prob  (all droped feature points)
    # block_size : drop掉的block大小
    def __init__(self, drop_rate=0.1, block_size=7):
        super(Drop, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
 
    def forward(self, x):
        if self.drop_rate == 0:
            return x
        # 设置gamma,比gamma小的设置为1,大于gamma的为0（得到丢弃比率的随机点个数）算法第五步
        # all droped feature center points
        gamma = self.drop_rate / (self.block_size**2)
        # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
        # 这里先生成有效范围大小的mask，随后通过零填充补充到原图大小，方便中心点的选取
        mask = (torch.rand(x.shape[0], x.shape[2]-self.block_size+1, x.shape[3]-self.block_size+1) < gamma).float()
        mask = F.pad(mask,(self.block_size//2,self.block_size//2,self.block_size//2,self.block_size//2))
        mask = mask.to(x.device)
        # fg1 = plt.figure()
        # fg1.add_subplot(2,2,1)
        # plt.imshow(mask[0],cmap='gray')
        # compute block mask
        block_mask = self._compute_block_mask(mask)
        # fg1.add_subplot(2,2,2)
        # plt.imshow(block_mask[0], cmap='gray')
        # fg1.add_subplot(2,2,3)
        # plt.imshow(x[0][0])
        # plt.show()

        # apply block mask,为算法图的第六步
        out = x * block_mask[:, None, :, :]
        # Normalize the features,对应第七步
        out = out * block_mask.numel() / block_mask.sum()
        return out
 
    def _compute_block_mask(self, mask):
        # 取最大值,这样就能够取出一个block的块大小的1作为drop,当然需要翻转大小,使得1为0,0为1
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

if __name__ == '__main__':
    test_data1 = torch.randn(2,3,256,256)
    test_model = Drop()
    test_data2 = test_model(test_data1)
    # print(test_data2)
    