import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,512,4,1,0,bias=False),  #often used for upsampling in GANs, 512-> no of feature maps or output channels, 4-> size of kernel(4*4), 1-> stride of convolution, 0->padding
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1,bias=False), #512-> no if input channels from he prev. layer, 256->  no of output channels, 4-> size of kernel, 2-> stride, 1->padding
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,3,4,2,1,bias=False), #3-> output channels representing RGB channels of an image
            nn.Tanh() #maps the output to range[-1,1]
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,128,4,2,1,bias=False), #often used for downsampling
            nn.LeakyReLU(0.2,inplace=True), # Applies the LeakyReLU activation function with a negative slope of 0.2. The operation is performed in-place to save memory.
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512,1,4,1,0,bias=False), #1 channel, producing a single output value indicating real or fake).
            nn.Sigmoid()
        )

    def forward(self,input):
        return self.main(input).view(-1,1)
    
# why leakyrelu
# Addressing the Dying ReLU Problem
# Stability in GAN Training(The non-zero gradient of LeakyReLU for negative inputs can contribute to more stable training dynamics by ensuring that gradients flow through all parts of the network.)
# Comparative Performance with Other Activation