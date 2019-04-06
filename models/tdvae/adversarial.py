import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Discriminator(torch.nn.Module):
    def __init__(self, channels, d_size=16):
        super().__init__()
        # Input_dim = channels (Cx224x160)
        # Output_dim = 1

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=channels, out_channels=d_size,
                                                      kernel_size=7, stride=2, padding=3))
        self.norm1 = nn.InstanceNorm2d(d_size, affine=True)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=d_size, out_channels=d_size*2,
                                                      kernel_size=3, stride=2, padding=1))
        self.norm2 = nn.InstanceNorm2d(d_size*2, affine=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(in_channels=d_size*2, out_channels=d_size*(2**2),
                                                      kernel_size=3, stride=2, padding=1))
        self.norm3 = nn.InstanceNorm2d(d_size*(2**2), affine=True)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(in_channels=d_size*(2**2), out_channels=d_size*(2**3),
                                                      kernel_size=3, stride=2, padding=1))
        self.norm4 = nn.InstanceNorm2d(d_size*(2**3), affine=True)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

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


# The following code is adapted with modifications from https://github.com/heykeetae/Self-Attention-GAN
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


def prime_facs(n):
    facs = []
    while n > 1:
        divisors = [i for i in range(2, n+1) if n % i == 0]
        fac = np.min(divisors)
        facs.append(fac)
        n = n//fac
    return sorted(facs, reverse=True)


class SAGANGenerator(nn.Module):
    """Generator."""
    def __init__(self, image_size=(3, 112, 80), z_dim=32, d_hidden=32):
        super(SAGANGenerator, self).__init__()
        layers = []

        x_strides = prime_facs(image_size[1])
        y_strides = prime_facs(image_size[2])
        while len(x_strides) < len(y_strides):
            x_strides.append(1)
        while len(y_strides) < len(x_strides):
            y_strides.append(1)

        dims = []

        layer1 = []
        mult = 2 ** (len(x_strides) - 3)  # 8
        layer1.append(nn.ConvTranspose2d(z_dim, d_hidden * mult, (x_strides[0], y_strides[0])))
        layer1.append(nn.BatchNorm2d(d_hidden * mult))
        layer1.append(nn.ReLU())
        layer1 = nn.Sequential(*layer1)
        layers.append(layer1)

        curr_dim = d_hidden * mult
        dims.append(curr_dim)
        for sx, sy in zip(x_strides[1:-1], y_strides[1:-1]):
            layer = []
            layer.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, (sx, sy), (sx//2, sy//2)))
            layer.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer.append(nn.ReLU())
            curr_dim = int(curr_dim / 2)
            layer.append(nn.Conv2d(curr_dim, curr_dim, 3, 1, 1))
            layer.append(nn.BatchNorm2d(int(curr_dim)))
            layer.append(nn.ReLU())

            dims.append(curr_dim)
            layers.append(layer)

        last = []
        last.append(nn.ConvTranspose2d(curr_dim, curr_dim, 4,
                                       (x_strides[-1], y_strides[-1]),
                                       (x_strides[-1]//2, y_strides[-1]//2)))
        last.append(nn.BatchNorm2d(int(curr_dim)))
        last.append(nn.ReLU())
        last.append(nn.Conv2d(curr_dim, 3, 3, 1, 1))
        last.append(nn.Sigmoid())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(dims[-2], 'relu')
        self.attn2 = Self_Attn(dims[-1], 'relu')
        layers[-2].append(self.attn1)
        layers[-1].append(self.attn2)

        self.layers = nn.ModuleList([nn.Sequential(*layer) for layer in layers])

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = z
        for i, layer in enumerate(self.layers):
            out = layer(out)

        out = self.last(out)
        return out

