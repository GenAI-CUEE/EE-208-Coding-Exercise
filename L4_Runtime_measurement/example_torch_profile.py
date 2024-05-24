import numpy as np
import time 
from datetime import timedelta 
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
import torchvision.transforms as transforms 


warmup_iters = 25


def gaussuian_kernel(kernel_size, sigma=1, muu=0): 
    # Initializing value of x,y as grid of kernel size  
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2) 
 
    # Calculating Gaussian filter
    gauss = (1/(2 * np.pi * sigma**2)) * np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) 

    return gauss



class CustomKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(CustomKernel, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel.shape, stride=1, padding=1)

        # Set the convolution weights using the custom kernel
        for i in range(out_channels):
            for j in range(in_channels):
                self.conv.weight.data[i, j, :, :] = torch.tensor(kernel, dtype=torch.float32)  # Apply the custom kernel 

        self.conv.bias.data.fill_(0)  # Set the bias to zero 
        
    def forward(self, x): 
        return self.conv(x)
    


im          = Image.open("Smurfs500.jpg") #These two lines
im_arr      = np.array(im)
image_sum   = im_arr.sum(axis=2)
im_arr      = image_sum/image_sum.max() 

h, w = im_arr.shape

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kernel = torch.tensor(gaussuian_kernel(3)).to(device)


Conv_Network = CustomKernel(in_channels=1, out_channels=1, kernel=kernel).to(device)
im_arr       = torch.tensor(im_arr.reshape(1,1,h,w), dtype=torch.float32).to(device) 


for i in range(50):
    if i == warmup_iters: 
        torch.cuda.cudart().cudaProfilerStart()

    if i >= warmup_iters:
        torch.cuda.nvtx.range_push("Conv2D")
        Conv_Network(im_arr.view(1,1,h,w))
        torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()