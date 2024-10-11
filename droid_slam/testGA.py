import torch
from gaussianMask_cuda import GaussianMask

# float dEXP_cov1 = exp_comp*0.5*(x1-mean_x)*(x1-mean_x)/(cov_1*cov_1);
#           float dP_cov1 = (dEXP_cov1*6.28*sqrt(cov_1*cov_2)-exp_comp*3.14*sqrt(cov_2/cov_1))/(6.28*6.28*cov_1*cov_2);
#           float dEXP_cov2 = exp_comp*0.5*(y1-mean_y)*(y1-mean_y)/(cov_2*cov_2);
#           float dP_cov2 = (dEXP_cov2*6.28*sqrt(cov_1*cov_2)-exp_comp*3.14*sqrt(cov_1/cov_2))/(6.28*6.28*cov_1*cov_2);

def custom_op(mean,cov,x,v):
    f1 = (x - mean) ** 2 / cov
    f1 = -0.5 * (f1[0]+f1[1])
    exp_comp = torch.exp(f1)
    # denom = 6.28 * torch.sqrt(cov[0] * cov[1])
    # pdf = exp_comp / denom
    v1 = v * exp_comp + v
    return v1
def custom_op_grad1(mean,cov,x,v):
    f1 = (x - mean)**2/cov
    f1 = -0.5 * (f1[0]+f1[1])
    denom = 6.28 * torch.sqrt(cov[0] * cov[1])
    exp_comp = torch.exp(f1)

    meanx_grad = v * exp_comp * (x[0]-mean[0])/cov[0]
    meany_grad = v * exp_comp * (x[1] - mean[1]) / cov[1]
    mean_grad = torch.Tensor([meanx_grad,meany_grad])
    return mean_grad

def custom_op_grad2(mean,cov,x,v):
    f1 = (x - mean) ** 2 / cov
    f1 = -0.5 * (f1[0] + f1[1])
    exp_comp = torch.exp(f1)
    denom = 6.28 * torch.sqrt(cov[0] * cov[1])

    dEXP_cov1 = v * exp_comp*0.5*((x[0] - mean[0])**2)/(cov[0]**2)
    dEXP_cov2 = v * exp_comp * 0.5 * ((x[1] - mean[1]) ** 2) / (cov[1] ** 2)
    # cov1_grad = v * (dEXP_cov1 * denom - exp_comp * 3.14 * torch.sqrt(cov[1]/cov[0])) / (denom**2)
    # cov2_grad = v * (dEXP_cov2 * denom - exp_comp * 3.14 * torch.sqrt(cov[0]/cov[1])) / (denom**2)
    cov_grad = torch.Tensor([dEXP_cov1, dEXP_cov2])
    return cov_grad

def check_gradient(custom_op,custom_op_grad,mean,cov,x,v,eps=1e-4):
    custom_gard = custom_op_grad(mean,cov,x,v)
    jgard = custom_gard[0]+custom_gard[1]

    diff_mean = mean.clone()
    diff_mean = diff_mean + eps
    f_plus = custom_op(diff_mean,cov,x,v)
    diff_mean = diff_mean - 2*eps
    f_minus = custom_op(diff_mean,cov,x,v)
    numerical_grad = (f_plus-f_minus)/(2*eps)

    relative_error = (numerical_grad-jgard).abs()/jgard.abs()
    sd = relative_error.max()
    return relative_error


v = torch.Tensor([3])
x = torch.Tensor([4,4])
mean = torch.Tensor([8,2])
cov = torch.Tensor([8,3])

check_gradient(custom_op,custom_op_grad1,mean,cov,x,v)

sd = custom_op(mean,cov,x,v)
print(1e-3)


