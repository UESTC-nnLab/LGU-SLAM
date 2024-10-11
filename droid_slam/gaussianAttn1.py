import torch
import torch.nn as nn
import math
import time
from droid_slam.modules.kan import KANLinear

class GaussianAttention(nn.Module):
    def __init__(self, h,w, dim=1,):
        super(GaussianAttention, self).__init__()
        self.covFilter = nn.Sequential(torch.nn.Conv2d(dim,dim,24,dilation=2),
                                       nn.ReLU(),
                                       nn.Linear(18,1))
        self.cov = torch.eye(2)

        coy, cox = torch.meshgrid(
            torch.arange(h).float(),
            torch.arange(w).float())
        coord = torch.stack([cox, coy], dim=-1)
        self.coord = coord.view(h * w, -1)

    def forward(self,x,h,w):
        b,n,c,s,_,_ = x.size()
        x1 = x.view(b * n * c, s, h * w).unsqueeze(dim=3)

        coords = torch.zeros_like(torch.cat((x1, x1), dim=3))

        x = x.permute(0,1,3,2,4,5).contiguous().view(b*n*s,c,h,w)

        x = self.covFilter(x).view(b*n,s,c,2,1)
        # torch.cuda.empty_cache()
        x = x.permute(0, 2, 1, 3, 4)
        x = x.view(b * n * c, s, 2)
        x = torch.sigmoid(x)
        x = torch.nn.functional.normalize(x, dim=2) * 5 + 1

        # x3 = torch.unsqueeze(x,dim=2)
        covn = torch.zeros_like(torch.cat((torch.unsqueeze(x,dim=2),torch.unsqueeze(x,dim=2)),dim=2))
        covn[:, :] = self.cov

        icov = torch.zeros_like(x)
        covn[:, :, 0, 0] = x[:, :, 0]
        covn[:, :, 1, 1] = x[:, :, 1]
        icov[:, :, 0] = 1 / x[:, :, 0]
        icov[:, :, 1] = 1 / x[:, :, 1]
        det = (x[:, :, 0] * x[:, :, 1])

        mean = torch.zeros_like(x)

        mean[:, 0:h * w, 0:2] = self.coord

        coords[:, :, 0:h * w, 0:2] = self.coord

        # diff = torch.ones_like(coords).cuda()
        # for i in range(b * n * c):
        #     for j in range(s):
        #         diff[i, j, :, :] = coords[i, j, :, :] - mean[i, j, :]

        mean = mean.unsqueeze(dim=2)

        diff = coords-mean

        icov = icov.unsqueeze(dim=2)

        temp = diff*icov

        exp_comp = temp*diff
        exp_comp = -0.5 *(exp_comp[:,:,:,0]+exp_comp[:,:,:,1])

        numerator = torch.exp(exp_comp)

        denominator = ((2 * math.pi) ** (2 / 2) * det).view(b*n*c, s, 1)

        pdf = numerator / denominator

        pdf = torch.nn.functional.normalize(pdf, dim=2)
        # t1 = pdf[:, :, 0].view(48, 64)
        # t2 = pdf[:, :, 1].view(48, 64)
        # t3 = pdf[:, :, 63].view(48, 64)
        return pdf


# GA = GaussianAttention(dim=1).cuda()
# total_params = sum(p.numel() for p in GA.parameters() if p.requires_grad)
#
if __name__ =="__main__":

    # covFilter = nn.Sequential(torch.nn.Conv2d(1, 1, 24, dilation=2),
    #                                nn.ReLU(),
    #                                nn.Linear(18, 1)).cuda()
    GA = GaussianAttention(48,64, dim=1).cuda()

    x = torch.ones((1, 12, 1, 48*64, 48, 64)).cuda()
    # start = time.time()
    # c = torch.zeros_like(x)
    # end = time.time()
    # print(end - start)


    for i in range(10000):
        start = time.time()
        x1 = GA(x,48,64)
        # torch.cuda.empty_cache()
        end = time.time()
        print(end-start)





# y = GA(x,48,64).view(1,5,1,3072,48,64)*x+x
