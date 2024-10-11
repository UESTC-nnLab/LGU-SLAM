import torch
import torch.nn as nn
import math
import time
import defCorrSample

class GaussianMaskCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mean, cov, corr, radius):
        mean = mean.float()
        cov = cov.float()
        ctx.save_for_backward(mean, cov, corr)
        ctx.radius = radius
        corr1, = defCorrSample.gaussianMask(mean,cov,corr,radius)
        return corr1

    @staticmethod
    def backward(ctx, grad_output):
        mean, cov, corr = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        means_grad,covs_grad = defCorrSample.gaussianMask_backward(mean, cov, corr, grad_output, ctx.radius)
        return means_grad, covs_grad, None, None

def per_Corr_Normalization(x, normalIndex, eps=1e-5):
    mean = torch.mean(x, dim=normalIndex)
    mean = mean.unsqueeze(dim=normalIndex[0]).unsqueeze(dim=normalIndex[1]).unsqueeze(dim=normalIndex[2])
    var = torch.var(x, dim=normalIndex, unbiased=False)+eps
    var = torch.sqrt(var).unsqueeze(dim=normalIndex[0]).unsqueeze(dim=normalIndex[1]).unsqueeze(dim=normalIndex[2])
    t = x - mean
    t = t / var
    return t

class GaussianMask(nn.Module):
    def __init__(self, h,w,):
        super(GaussianMask, self).__init__()
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

        coy, cox = torch.meshgrid(
            torch.arange(h).float(),
            torch.arange(w).float())
        coord = torch.stack([cox, coy], dim=-1)
        self.coord = coord.view(h , w, -1)

    def forward(self,x, corr):
        b,h,w,_ = x.size()

        tt = self.mapA(x)

        mean_ofs = self.meanMap(tt).view(b, h * w, 2)

        x = self.covMap(tt).view(b, h * w, 2)
        x = per_Corr_Normalization(x, [1, 2])

        x = torch.sigmoid(x) * 5 + 0.05
        det = (x[:, :, 0] * x[:, :, 1])
        x = x.view(-1,h,w,2).float()
        mean_ofs = mean_ofs.view(-1,h,w,2)

        mean = torch.zeros_like(x)
        mean[:, 0:h, 0:w, 0:2] = self.coord
        mean = mean + mean_ofs
        corr1 = GaussianMaskCuda.apply(mean,x,corr,6)
        denominator = 6.28 * torch.sqrt(det).view(b, h, w, 1,1)
        corr1 = corr1/denominator+corr

        return corr1,mean,det

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
    #                            nn.Linear(18, 1)).cuda()

    GA = GaussianMask(6,8).cuda()
    # total_params = sum(p.numel() for p in GA.parameters() if p.requires_grad)
    # print(total_params)
    x1 = torch.randn((1,6,8,256)).cuda()
    corr = torch.randn((1,6,8,6,8)).cuda()
    ba = []
    for i in range(12):
        start = time.time()
        ba.append(GA(x1,corr))
        end = time.time()
        print(end-start)

print()



# y = GA(x,48,64).view(1,5,1,3072,48,64)*x+x
