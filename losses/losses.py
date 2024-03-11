'''
@copyright ziqi-jin
You can create custom loss function in this file, then import the created loss in ./__init__.py and add the loss into AVAI_LOSS
'''
import torch.nn as nn
import torch


# example
class CustormLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, y):
        pass


class DiceLoss(nn.Module):
    ''' only for binary classification '''
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, x, y):
        ''' x: input 1ch float, y: target int '''
        x_prob = torch.sigmoid(x)

        intersection = (x_prob * y).sum()

        dice = 1 - (2 * intersection + self.smooth) / (x_prob.sum() + y.sum() + self.smooth)

        return dice


