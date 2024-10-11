import torch
import torch.nn as nn
from droid_slam.modules.kan import KANLinear
class Kanbias_GRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128):
        super(Kanbias_GRU, self).__init__()
        self.do_checkpoint = False

        self.dimAlignment = nn.Sequential(
            nn.Conv2d(i_planes, h_planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(h_planes),
            nn.GELU(),
        )
        self.crossDefAttn_z = nn.MultiheadAttention(h_planes,2)
        self.crossDefAttn_r = nn.MultiheadAttention(h_planes,2)

        self.convq = nn.Conv2d(h_planes*2, h_planes, 3, padding=1)

        self.kanz_glo = KANLinear(128, 128,grid_size=3)
        self.kanr_glo = KANLinear(128, 128,grid_size=3)
        self.kanq_glo = KANLinear(128, 128,grid_size=3)

        self.w = nn.Conv2d(h_planes, h_planes, 1, padding=0)


    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)

        inp = self.dimAlignment(inp)

        b, c, h, w = net.shape
        glo = torch.sigmoid(self.w(net)) * net
        glo = glo.view(b, c, h*w).mean(-1).view(b, c)

        kanz_glo = self.kanz_glo(glo).view(b, c,1,1)
        kanr_glo = self.kanr_glo(glo).view(b, c,1,1)
        kanq_glo = self.kanq_glo(glo).view(b, c,1,1)

        net1 = net.view(b,128,-1).contiguous().permute(0,2,1)
        inp1 = inp.view(b,128,-1).contiguous().permute(0,2,1)

        z = self.crossDefAttn_z(net1,inp1,inp1)[0].permute(0,2,1).contiguous().view(b,128,h,w)
        r = self.crossDefAttn_r(net1,inp1,inp1)[0].permute(0,2,1).contiguous().view(b,128,h,w)

        z = torch.sigmoid(z + kanz_glo)
        r = torch.sigmoid(r + kanr_glo)
        q = torch.tanh(self.convq(torch.cat([r * net, inp], dim=1)) + kanq_glo)

        net = (1-z) * net + z * q #
        return net


if __name__ =="__main__":
    gru = Kanbias_GRU(128,310)
    b = torch.randn((2,3072,128))
    crossDefAttn_z = nn.MultiheadAttention(512, 4)
    c = crossDefAttn_z(b,b,b)
    z = nn.Conv2d(128, 128, 3, padding=1)
    total_params = sum(p.numel() for p in crossDefAttn_z.parameters() if p.requires_grad)
    inp = torch.randn((36,310,40,64))
    net = torch.randn((36, 128, 40, 64))
    s = gru(net,inp)
    print()