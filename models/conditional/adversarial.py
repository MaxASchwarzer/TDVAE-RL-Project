import torch
from torch import nn
from torch.nn import functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, channels, d_size=16):
        super().__init__()
        # Filters [32, 64, 128, 128, 128]
        # Input_dim = channels (Cx224x160)
        # Output_dim = 1

        # Image (Cx32x32)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=channels, out_channels=d_size,
                                                      kernel_size=7, stride=2, padding=3))
        self.norm1 = nn.InstanceNorm2d(d_size, affine=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        # State (256x16x16)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=d_size, out_channels=d_size*2,
                                                      kernel_size=3, stride=2, padding=1))
        self.norm2 = nn.InstanceNorm2d(d_size*2, affine=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        # State (512x8x8)
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channels=d_size*2, out_channels=d_size*(2**2),
                                                      kernel_size=3, stride=2, padding=1))
        self.norm3 = nn.InstanceNorm2d(d_size*(2**2), affine=True)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        # output of main module --> State (1024x4x4)

        # Image (Cx32x32)
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(in_channels=d_size*(2**2), out_channels=d_size*(2**3),
                                                      kernel_size=3, stride=2, padding=1))
        self.norm4 = nn.InstanceNorm2d(d_size*(2**3), affine=True)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        # Image (Cx32x32)
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(in_channels=d_size*(2**3), out_channels=d_size*(2**4),
                                                      kernel_size=3, stride=2, padding=1))
        self.norm5 = nn.InstanceNorm2d(d_size*(2**4), affine=True)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
        self.final = nn.Conv2d(in_channels=d_size*(2**4), out_channels=1, kernel_size=(7, 5), stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        x2 = self.relu2(x2)

        x3 = self.conv3(x2)
        x3 = self.norm3(x3)
        x3 = self.relu3(x3)

        x4 = self.conv4(x3)
        x4 = self.norm4(x4)
        x4 = self.relu4(x4)

        x5 = self.conv5(x4)
        x5 = self.norm5(x5)
        x5 = self.relu5(x5)
        output = self.final(x5)
        return output

    def calc_loss(self, real, fake):
        r_out = self(real)
        f_out = self(fake)
        d_loss = -r_out.mean() + f_out.mean()
        g_loss = -f_out.mean() + r_out.mean()

        # Compute gradient penalty
        alpha = torch.rand(real.size(0), 1, 1, 1).cuda().expand_as(real)
        interpolated = (alpha * real + (1 - alpha) * fake)
        out = self(interpolated)

        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
        print((r_out.mean() - f_out.mean()).item(), d_loss_gp.item())

        # Backward + Optimize
        d_loss = 10*d_loss_gp + d_loss

        return d_loss, g_loss

