import torch
import torch.nn as nn
import math
from droid_slam.attn.normalization import InstanceL2Norm
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.t_pos1 = nn.Parameter(torch.randn(1, 1, 128))
        self.t_pos2 = nn.Parameter(torch.randn(1, 1, 128))
        norm_scale = math.sqrt(1.0 / (128 * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)
        self.W_qk = nn.Linear(128, 32)
        # self.W_v = nn.Linear(128, 128)


    def forward(self,f_list1, f_list2):
        b, n, c, h, w = f_list1.shape

        f_l1 = f_list1.view(b*n, c, h * w).contiguous().permute(0, 2, 1) + self.t_pos1
        f_l2 = f_list2.view(b*n, c, h * w).contiguous().permute(0, 2, 1) + self.t_pos2

        f_q = self.W_qk(f_l1)
        f_k = self.W_qk(f_l2)
        f_v = f_l2

        f_q = F.normalize(f_q, p=2, dim=-1)
        f_k = F.normalize(f_k, p=2, dim=-1)

        attn_Mat = torch.matmul(f_q, f_k.transpose(1, 2))

        attn_Mat = F.softmax(attn_Mat * 30, dim=-1)

        f_r = torch.bmm(attn_Mat, f_v)

        f_r = f_r+f_l2

        f_r = f_r.permute(0, 2, 1).contiguous().view(-1, c, h, w)
        f_r = self.norm(f_r)

        return f_r