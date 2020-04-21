import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class DropBlock2d(nn.Module):

    def __init__(self, keep_prob, block_size):
        super(DropBlock2d, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size


    def forward(self,x):

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        bsize, channels, feat_size, _ = x.shape

        if not self.training or self.keep_prob == 1.:
            return x
        else:

            gamma = self._compute_gamma(feat_size)

            block_mask = self._compute_block_mask(bsize, channels,feat_size,gamma)
            block_mask = block_mask.to(x.device)
            out = x * block_mask.float()
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_gamma(self, feat_size):

        gamma = ((1 - self.keep_prob)/(self.block_size ** 2))*((feat_size ** 2)/((feat_size - self.block_size +1 ) ** 2))
        # gamma = (1 - self.keep_prob)/(self.block_size ** 2)
        return gamma
 
    def _compute_block_mask(self,bsize, channels, feat_size,gamma):

        mask = (np.random.rand(bsize,channels,(feat_size - self.block_size + 1),(feat_size - self.block_size + 1)) < gamma)
        # print (mask)
        cods = np.where(mask == 1)
        assert len(cods) == 4
        cods = np.dstack(cods).squeeze(0)
        block_mask = np.ones(shape=(bsize,channels,feat_size,feat_size))
        for cod in cods :
            block_mask[cod[0],cod[1],cod[2]:cod[2]+self.block_size,cod[3]:cod[3]+self.block_size]=0

        # print (block_mask)
        return torch.from_numpy(block_mask)


class DropBlock2D_modifed(nn.Module):

    def __init__(self, keep_prob, block_size):
        super(DropBlock2D_modifed, self).__init__()

        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.keep_prob == 1.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(*x.shape) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask
        return block_mask

    def _compute_gamma(self, x):
        return (1 - self.keep_prob) / (self.block_size ** 2)


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.keep_prob = self.drop_values[self.i]
        self.i += 1


if __name__ == '__main__':
    dropblock = DropBlock2D_modifed(keep_prob=0.9, block_size=3)
    gamma = dropblock._compute_gamma(5)
    print(gamma)
    mask = (torch.rand(2,2,5,5) < gamma).float()
    print(mask)
    dropblock_mask = dropblock._compute_block_mask(mask)
    print(dropblock_mask.shape)