import cv2
import torch
import torch.nn as nn
import math
import time
import numpy as np
from droid_slam.modules.kan import KANLinear
from visualRes.visualGS import visualize_gs

class GaussianAttention(nn.Module):
    def __init__(self, h,w,):
        super(GaussianAttention, self).__init__()
        self.meanMap = nn.Linear(16, 2)
        self.covMap = nn.Linear(16, 2)
        self.map = nn.Linear(256, 16)
        self.cov = torch.eye(2)

        for m in self.meanMap.modules():
            m.weight.data.zero_()
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.covMap.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.map.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        self.mapA = nn.Sequential(self.map,nn.Tanh())
        self.count = 0
        coy, cox = torch.meshgrid(
            torch.arange(h).float(),
            torch.arange(w).float())
        coord = torch.stack([cox, coy], dim=-1)
        self.coord = coord.view(h * w, -1)

    def forward(self,x):
        b,h,w,_ = x.size()

        coords = torch.zeros((b, h * w, h * w)).cuda()

        coords = torch.cat((coords.unsqueeze(dim=3),coords.unsqueeze(dim=3)),dim=3)

        tt = self.mapA(x)

        mean_ofs = self.meanMap(tt).view(b,h*w,2)

        x = self.covMap(tt).view(b,h*w,2)
        x = per_Corr_Normalization(x,[1,2])
        x = torch.sigmoid(x) * 5 + 0.05

        icov = torch.zeros_like(x)
        icov[:, :, 0] = 1 / x[:, :, 0]
        icov[:, :, 1] = 1 / x[:, :, 1]
        det = (x[:, :, 0] * x[:, :, 1])
        # det1 = icov[:, :, 0]*icov[:, :, 1]

        mean = torch.zeros_like(x)

        mean[:, 0:h * w, 0:2] = self.coord

        mean = mean + mean_ofs

        coords[:, :, 0:h * w, 0:2] = self.coord

        mean = mean.unsqueeze(dim=2)

        diff = coords-mean

        icov = icov.unsqueeze(dim=2)

        temp = diff*icov

        exp_comp = temp*diff
        exp_comp = -0.5 *(exp_comp[:,:,:,0]+exp_comp[:,:,:,1])

        numerator = torch.exp(exp_comp)


        denominator = 6.28*torch.sqrt(det).view(b, h * w, 1)

        pdf = (numerator / denominator)*3

        # img = cv2.imread("/home/honsen/asd/qwe/000026_left.png")
        # img = cv2.resize(img,(512,384))
        # vdet = det1.view(48,64)*14
        # vdet = vdet.cpu().numpy()
        # vdet = np.uint8(vdet)
        # vdet = cv2.resize(vdet, (512, 384))
        # vdet = cv2.applyColorMap(vdet, cv2.COLORMAP_JET)
        # fus1 = cv2.addWeighted(img, 0.5, vdet, 0.5, 0)
        # cv2.imshow("visualf1", fus1)
        # cv2.waitKey(3000)
        # gs_kernel = pdf[0, :, :]*4
        # gs_kernel = gs_kernel.cpu().numpy()
        # #
        # visualize_gs(self.count,1, gs_kernel)
        # visualize_gs(self.count+1,0, vdet)
        # self.count+=1
        return pdf,mean,det

def per_Corr_Normalization(x, normalIndex, eps=1e-5):
    mean = torch.mean(x, dim=normalIndex)
    mean = mean.unsqueeze(dim=1).unsqueeze(dim=2)
    var = torch.var(x, dim=normalIndex, unbiased=False)+eps
    var = torch.sqrt(var).unsqueeze(dim=1).unsqueeze(dim=2)
    t = x - mean
    t = t / var
    return t

# GA = GaussianAttention(dim=1).cuda()
# total_params = sum(p.numel() for p in GA.parameters() if p.requires_grad)
#
if __name__ =="__main__":

    # covFilter = nn.Sequential(torch.nn.Conv2d(1, 1, 24, dilation=2),
    #                                nn.ReLU(),
    #                                nn.Linear(18, 1)).cuda()
    GA = GaussianAttention(48,64).cuda()
    # total_params = sum(p.numel() for p in GA.parameters() if p.requires_grad)
    # print(total_params)
    x1 = torch.randn((12,48,64,256)).cuda()

    ba = []
    for i in range(12):
        start = time.time()
        ba.append(GA(x1))
        torch.cuda.empty_cache()
        end = time.time()
        print(end-start)

print()



# y = GA(x,48,64).view(1,5,1,3072,48,64)*x+x
