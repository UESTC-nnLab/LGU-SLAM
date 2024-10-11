import time

import cv2
import cv2 as cv
import numpy
import torch
import torch.nn.functional as F
import defCorrSample
import defCorrSample1
import droid_backends
from torch.autograd import gradcheck
import droid_test
from visualRes.visualGS import visualize_gs
class CorrSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = defCorrSample.corr_index_forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = defCorrSample.corr_index_backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None

class DefCorrSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, volume, coords, offset, radius):
        offset = offset.float()
        volume = volume.float()
        ctx.save_for_backward(volume,coords,offset)
        ctx.radius = radius
        corr, = defCorrSample.defCorr_index_forward(volume, coords, offset, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords, offset = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume,offset_grad = defCorrSample.defCorr_index_backward(volume, coords, offset, grad_output, ctx.radius)
        return grad_volume, None, offset_grad, None

def per_Corr_Normalization(x, normalIndex, eps=1e-5):
    mean = torch.mean(x, dim=normalIndex)
    mean = mean.unsqueeze(dim=normalIndex[0]).unsqueeze(dim=normalIndex[1]).unsqueeze(dim=normalIndex[2])
    var = torch.var(x, dim=normalIndex, unbiased=False)+eps
    var = torch.sqrt(var).unsqueeze(dim=normalIndex[0]).unsqueeze(dim=normalIndex[1]).unsqueeze(dim=normalIndex[2])
    t = x - mean
    t = t / var
    return t
class CorrBlock:
    def __init__(self,beta, ofsMap, ofs_residual, GA, fmap1, fmap2, num_levels=4, radius=3):

        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.GA = GA
        self.ofsMap = ofsMap
        self.beta = beta
        # self.bias = bias
        self.ofs_residual = ofs_residual
        corr = CorrBlock.corr(fmap1,fmap2)
        # all pairs correlation
        b, n, c, s, h, w = corr.shape
        # corr = corr.view(b*n*c,h,w,h,w).float()

        self.t = torch.cat((fmap1.view(b * n, 128, h, w),
                       fmap2.view(b * n, 128, h, w)), dim=1)#.permute(0,2,3,1).contiguous()

        self.offset = []

        self.fpn_offset_generate(self.t)
        #
        # self.t = self.t.permute(0,2,3,1).contiguous()
        # corr,mean_n,det = self.GA(self.t,corr)

        # corrn = None
        # mean_n = []
        # dets = []
        # for i in range(n):
        #     corri = corr[:, i, :, :, :, :].unsqueeze(dim=1)
        #     # ct = corri.view(3072,3072)
        #     # visualize_gs(3,1,ct.cpu().numpy())
        #     temp, mean,det = self.GA(self.t.view(b, n, h, w, 128 * 2)[:, i, :, :, :])
        #     corri = temp.view(b, 1, c, s, h, w) * corri + corri
        #     dets.append(det)
        #     mean_n.append(mean)
        #     # ct = corri.view(3072, 3072)
        #     # visualize_gs(4, 1,ct.cpu().numpy())
        #     if i == 0:
        #         corrn = corri
        #     else:
        #         corrn = torch.cat((corrn, corri), dim=1)
        # corr = corrn
        #
        # mean_n = torch.cat((mean_n),dim=0)
        # det = torch.cat((dets),dim=0)
        #
        # self.mean_n = mean_n.view(b,n,h,w,2)
        # self.theta = 2*det.view(b,n,h,w)

        corr = corr.view(b,n*c,h,w,h,w)
        batch, num, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(batch*num*h1*w1, 1, h2, w2)

        for i in range(self.num_levels):
            self.corr_pyramid.append(
                corr.view(batch*num, h1, w1, h2//2**i, w2//2**i))
            corr = F.avg_pool2d(corr, 2, stride=2)

    def __call__(self, coords):
        out_pyramid = []
        batch, num, ht, wd, _ = coords.shape
        coords = coords.permute(0,1,4,2,3)
        coords = coords.contiguous().view(batch*num, 2, ht, wd)

        # corr_ = CorrSampler.apply(self.corr_pyramid[1], coords/2, 2)
        # corr_ = corr_.permute(0,3,4,1,2)
        # corrUncertain = torch.var(corr_,dim=[3,4])
        # corrUncertain_mask = torch.sigmoid(corrUncertain).view(batch*num, ht, wd, 1)


        # visualf1 = numpy.uint8(255 * corrUncertain_mask.view(ht,wd).cpu().numpy())
        # visualf1 = cv2.resize(visualf1, (512, 384))
        # visualf1 = cv2.applyColorMap(visualf1, cv2.COLORMAP_JET)
        # fus1 = cv2.addWeighted(self.img2, 0.5, visualf1, 0.5, 0)
        # cv.imshow("visualf1", fus1)
        # cv.waitKey(3000)
        # visualize_gs(1,0,corrUncertain_mask.view(ht,wd).cpu().numpy()*255)
        # visualize_gs(2,self.corr_pyramid[0][0,9,33,:,:].cpu().numpy())

        # self.offset[1] = self.offset[1]*corrUncertain_mask

        for i in range(self.num_levels):
            # corr = CorrSampler.apply(self.corr_pyramid[i], coords/2**i, self.radius)
            corr = DefCorrSampler.apply(self.corr_pyramid[i], coords / 2 ** i, self.offset[i].contiguous().view(batch*num, ht, wd, 2*self.radius+1, 2*self.radius+1, 2), self.radius)
            out_pyramid.append(corr.view(batch, num, -1, ht, wd))

        return torch.cat(out_pyramid, dim=2)#, self.mean_n, self.theta

    def cat(self, other):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = torch.cat([self.corr_pyramid[i], other.corr_pyramid[i]], 0)
            self.offset[i] = torch.cat([self.offset[i], other.offset[i]], 0)
        return self

    def fpn_offset_generate(self, t):
        b,c,h,w = t.size()

        self.offset.append(self.ofsMap(t))
        t1 = F.avg_pool2d(t, kernel_size=2, stride=2)
        self.offset.append(self.ofs_residual(t1))

        self.offset[1] = F.interpolate(self.offset[1], (h, w))

        self.offset[0] = F.tanh(per_Corr_Normalization(self.offset[0], [1,2,3]))*4
        self.offset[1] = (F.tanh(per_Corr_Normalization(self.offset[1], [1,2,3]))*4+ self.offset[0])/2

        self.offset[0] = self.offset[0].permute(0, 2, 3, 1)
        self.offset[1] = self.offset[1].permute(0, 2, 3, 1)

        self.offset.append(torch.zeros_like(self.offset[0]))
        self.offset.append(torch.zeros_like(self.offset[0]))
        self.offset[2] = self.offset[2].detach()
        self.offset[3] = self.offset[3].detach()

    def __getitem__(self, index):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = self.corr_pyramid[i][index]
            self.offset[i] = self.offset[i][index]
        return self


    @staticmethod
    def corr(fmap1, fmap2):
        """ all-pairs correlation """
        batch, num, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch*num, dim, ht*wd) / 4.0
        fmap2 = fmap2.reshape(batch*num, dim, ht*wd) / 4.0
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        return corr.view(batch, num, 1,  ht*wd, ht, wd)


class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, r):
        ctx.r = r
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr, = droid_backends.altcorr_forward(fmap1, fmap2, coords, ctx.r)
        return corr

    @staticmethod
    def backward(ctx, grad_corr):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_corr = grad_corr.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            droid_backends.altcorr_backward(fmap1, fmap2, coords, grad_corr, ctx.r)
        return fmap1_grad, fmap2_grad, coords_grad, None


class AltCorrBlock:
    def __init__(self, beta, ofsMap, ofs_residual, GA, fmaps, num_levels=4, radius=3):
        self.num_levels = num_levels
        self.radius = radius
        self.GA = GA
        self.ofsMap = ofsMap
        self.beta = beta
        self.ofs_residual = ofs_residual
        self.offset = []

        B, N, C, H, W = fmaps.shape
        fmaps = fmaps.view(B*N, C, H, W) / 4.0
        
        self.pyramid = []
        for i in range(self.num_levels):
            sz = (B, N, H//2**i, W//2**i, C)
            fmap_lvl = fmaps.permute(0, 2, 3, 1).contiguous()
            self.pyramid.append(fmap_lvl.view(*sz))
            fmaps = F.avg_pool2d(fmaps, 2, stride=2)
  
    def corr_fn(self, coords, ii, jj):
        B, N, H, W, S, _ = coords.shape
        coords = coords.permute(0, 1, 4, 2, 3, 5)

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][:, ii]
            fmap2_i = self.pyramid[i][:, jj]

            coords_i = (coords / 2**i).reshape(B*N, S, H, W, 2).contiguous()
            fmap1_i = fmap1_i.reshape((B*N,) + fmap1_i.shape[2:])
            fmap2_i = fmap2_i.reshape((B*N,) + fmap2_i.shape[2:])

            corr, = droid_backends.altcorr_forward(fmap1_i.float(), fmap2_i.float(), coords_i, self.radius)
            corr = corr.view(B, N, S, -1, H, W).permute(0, 1, 3, 4, 5, 2)
            corr_list.append(corr)

        corr = torch.cat(corr_list, dim=2)
        return corr

    def corr_fn1(self, coords, ii, jj):
        B, N, H, W, S, _ = coords.shape
        coords = coords.permute(0, 1, 4, 2, 3, 5)
        fmap1 = self.pyramid[0][:, ii]*4.0
        fmap2 = self.pyramid[0][:, jj]*4.0
        fmap1 = fmap1.reshape((B * N,) + fmap1.shape[2:])
        fmap2 = fmap2.reshape((B * N,) + fmap2.shape[2:])
        _, h1, w1, C = fmap2.shape

        fmap1 = fmap1.permute(0,3,1,2)
        fmap2 = fmap2.permute(0, 3, 1, 2)

        t = torch.cat((fmap1,
                       fmap2), dim=1).float()
        self.offset = []
        self.offset_generate(t)

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius

            fmap1_i = self.pyramid[0][:, ii]
            fmap2_i = self.pyramid[i][:, jj]

            coords_i = (coords / 2 ** i).reshape(B * N, S, H, W, 2).contiguous()
            fmap1_i = fmap1_i.reshape((B * N,) + fmap1_i.shape[2:])
            fmap2_i = fmap2_i.reshape((B * N,) + fmap2_i.shape[2:])

            corr, = defCorrSample1.lowMem_defSample(fmap1_i.float(), fmap2_i.float(), coords_i, self.offset[i].contiguous().view(B * N, H, W, (2 * self.radius + 1) , (2 * self.radius + 1), 2).float(), self.radius)

            corr = corr.view(B, N, S, -1, H, W).permute(0, 1, 3, 4, 5, 2)
            corr_list.append(corr)

        corr = torch.cat(corr_list, dim=2)
        return corr

    def offset_generate(self, t):
        b, c, h, w = t.size()

        self.offset.append(self.ofsMap(t))
        t1 = F.avg_pool2d(t, kernel_size=2, stride=2)
        self.offset.append(self.ofs_residual(t1))

        self.offset[1] = F.interpolate(self.offset[1], (h, w))

        self.offset[0] = F.tanh(per_Corr_Normalization(self.offset[0], [1, 2, 3])) * 4
        self.offset[1] = (F.tanh(per_Corr_Normalization(self.offset[1], [1, 2, 3])) * 4 + self.offset[0]) / 2

        self.offset[0] = self.offset[0].permute(0, 2, 3, 1)
        self.offset[1] = self.offset[1].permute(0, 2, 3, 1)

        self.offset.append(torch.zeros_like(self.offset[0]))
        self.offset.append(torch.zeros_like(self.offset[0]))
        self.offset[2] = self.offset[2].detach()
        self.offset[3] = self.offset[3].detach()


    def __call__(self, coords, ii, jj):
        squeeze_output = False
        if len(coords.shape) == 5:
            coords = coords.unsqueeze(dim=-2)
            squeeze_output = True

        corr = self.corr_fn1(coords, ii, jj)
        
        if squeeze_output:
            corr = corr.squeeze(dim=-1)

        return corr.contiguous()

if __name__ =="__main__":
    torch.nn.BatchNorm2d()
    x=torch.randn(4,requires_grad=True)
    a  = torch.ones((1,3,4,3,4))
    a = a.cuda()
    a.requires_grad = True

    coy, cox = torch.meshgrid(
        torch.arange(3).float(),
        torch.arange(4).float())
    coord = torch.stack([cox, coy], dim=-1)
    b = coord.permute(2,0,1).unsqueeze(dim=0).contiguous().cuda()
    b.requires_grad = True
    c = torch.randn(1,3,4,3,3,2).cuda()
    c.requires_grad = True
    # s = gradcheck(torch.exp, (x,),eps=0.0001)
    # dd, = CorrSampler.apply(a,b,2)
    # v = torch.nonzero(dd)
    s = gradcheck(CorrSampler.apply,(a,b,1),eps=3)
    print(s)