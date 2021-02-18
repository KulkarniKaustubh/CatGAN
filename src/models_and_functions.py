import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on GPU')
else:
    device = 'cpu'
    print('Running on CPU')

# GENERATOR CLASS

class Generator(nn.Module):
    def __init__(self, noise_chan=100, im_chan=3):
        super(Generator, self).__init__()
        
        self.noise_chan = noise_chan
        
        self.gen = nn.Sequential(
            self.intermediate_block(noise_chan, 64*16, kernel_size=4, stride=1),  # inp 1x1 kernel 4x4 op 4x4x1024  [output = (inp-1)*stride + kernel]
            self.intermediate_block(64*16, 64*8, kernel_size=2, stride=2),   # (4-1)*2 + 2 = 8x8x512 image
            self.intermediate_block(64*8, 64*4, kernel_size=2, stride=2),    # (8-1)*2 + 2 = 16x16x256 image
            self.intermediate_block(64*4, 64*2, kernel_size=2, stride=2),    # 32x32x128 output 
            self.final_block(64*2, im_chan, kernel_size=2, stride=2),        # 64x64x3 output
        )
        
    def intermediate_block(self, input_channels, output_channels, kernel_size, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def final_block(self, input_channels, output_channels, kernel_size, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.Tanh()
        )
    
    def forward(self, noise):
        noise = noise.view(len(noise), self.noise_chan, 1, 1)    # PyTorch takes (batch, channels, height, width)
        return self.gen(noise)
    

# Function to get noise
def get_noise(n_samples, noise_chan, device=device):
    return torch.randn(n_samples, noise_chan, device=device)


# DISCRIMINATOR CLASS

class Discriminator(nn.Module):
    def __init__(self, im_chan=3):
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential(
            self.intermediate_block(im_chan, 16, kernel_size=4, stride=2),
            self.intermediate_block(16, 16*2, kernel_size=4, stride=2),
            self.intermediate_block(16*2, 16*4, kernel_size=4, stride=2),
            self.final_block(16*4, 1, kernel_size=4, stride=2),
        )
        
    def intermediate_block(self, input_channels, output_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    
    def final_block(self, input_channels, output_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
        )
    
    def forward(self, img):
        return self.disc(img)