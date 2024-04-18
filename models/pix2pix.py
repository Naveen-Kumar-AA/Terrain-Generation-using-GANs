# import torch.nn as nn
# import torch 


# class Pix2PixGenerator(nn.Module):
#   def __init__(self, in_channels=1, out_channels=3, features=32):
#     super(Pix2PixGenerator, self).__init__()

#     # Downward (encoder) path with convolutional and pooling layers
#     self.down1 = nn.Sequential(
#         nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.1)
#     )
#     self.pool1 = nn.MaxPool2d(2, stride=2)

#     self.down2 = nn.Sequential(
#         nn.Conv2d(features, features * 2, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True)
#     )
#     self.pool2 = nn.MaxPool2d(2, stride=2)

#     self.down3 = nn.Sequential(
#         nn.Conv2d(features * 2, features * 3, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True)
#     )
#     self.pool3 = nn.MaxPool2d(2, stride=2)

#     self.down4 = nn.Sequential(
#         nn.Conv2d(features * 3, features * 3, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True)
#     )
#     self.pool4 = nn.MaxPool2d(2, stride=2)

#     # Upward (decoder) path with transposed convolution and concatenation
#     self.up5 = nn.Sequential(
#         nn.ConvTranspose2d(features * 3, features * 2, kernel_size=2, stride=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.3),
#         nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1),  # Concatenation layer
#         nn.ReLU(inplace=True)
#     )

#     self.up6 = nn.Sequential(
#         nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(features * 3, features, kernel_size=3, padding=1),  # Concatenation layer
#         nn.ReLU(inplace=True)
#     )

#     self.up7 = nn.Sequential(
#         nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(features * 2, features // 2, kernel_size=3, padding=1),  # Concatenation layer
#         nn.ReLU(inplace=True)
#     )

#     self.up8 = nn.Sequential(
#         nn.ConvTranspose2d(features // 2, features // 4, kernel_size=2, stride=2),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(features * 2, features // 4, kernel_size=3, padding=1),  # Concatenation layer
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.1)
#     )

#     self.up9 = nn.Sequential(
#         nn.ConvTranspose2d(features // 4, features // 8, kernel_size=2, stride=2),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(features // 2, features // 8, kernel_size=3, padding=1),  # Concatenation layer
#         nn.ReLU(inplace=True)
#     )

#     # Final output layer with tanh activation
#     self.final = nn.Conv2d(features // 8, out_channels, kernel_size=1)
#     self.tanh = nn.Sigmoid()

#   def forward(self, x):
#     # Pass through downsampling layers
#     d1 = self.down1(x)
#     p1 = self.pool1(d1)

#     d2 = self.down2(p1)
#     p2 = self.pool2(d2)

#     d3 = self.down3(p2)
#     p3 = self.pool3(d3)

#     d4 = self.down4(p3)
#     p4 = self.pool4(d4)

#     # Upward (decoder) path with transposed convolution and concatenation
#     up5 = self.up5(p4)  # Up-sample p4
#     up5_concat = torch.cat((up5, p3), dim=1)  # Concatenate up-sampled feature maps with p3, not d4
#     up6 = self.up6(up5_concat)  # Concatenation with d3 is already handled in the up6 layer
#     up7 = self.up7(up6)  # Concatenation with d2 is already handled in the up7 layer
#     up8 = self.up8(up7)  # Concatenation with d1 is already handled in the up8 layer

#     up9 = self.up9(up8)  # No concatenation here

#     output = self.tanh(self.final(up9))  # Apply tanh to final output
#     return output

# class Pix2PixDiscriminator(nn.Module):
#   def __init__(self, in_channels=6):  # Assuming input combines image and target
#     super(Pix2PixDiscriminator, self).__init__()

#     # PatchGAN architecture with convolutional layers and LeakyReLU activation
#     self.discriminator = nn.Sequential(
#       nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
#       nn.LeakyReLU(0.2),
#       nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#       nn.LeakyReLU(0.2),
#       nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#       nn.LeakyReLU(0.2),
#       nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
#       nn.BatchNorm2d(512),  # Batch normalization after LeakyReLU
#       nn.LeakyReLU(0.2)
#     )

#     # Final layer for classification with sigmoid activation
#     self.final = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
#     self.activation = nn.Sigmoid()  # Add sigmoid activation for 0-1 output

#   def forward(self, x):
#     # Pass through discriminator network
#     x = self.discriminator(x)
#     # Final output with sigmoid activation
#     output = self.activation(self.final(x))
#     return output

import torch.nn as nn
import torch

class DownSample(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(DownSample,self).__init__()
    # nn.Conv2d(in_channel, out_channel, kernel, stride, padding)
    self.model = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    down = self.model(x)
    return down 

   
class UpSample(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(UpSample,self).__init__()

    self.model = nn.Sequential(
      nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1, bias=False),
      nn.InstanceNorm2d(output_channels),
      nn.ReLU(inplace=True),
    )

  def forward(self, x, skip_input):
    x = self.model(x)
    x = torch.cat((x, skip_input), 1)
    return x
  
class Pix2PixGenerator(nn.Module):
  def __init__(self, in_channels=3, out_channels=3):
    super(Pix2PixGenerator, self).__init__()

    self.down1 = DownSample(in_channels, 64)
    self.down2 = DownSample(64,128)
    self.down3 = DownSample(128,256)
    self.down4 = DownSample(256,512)
    self.down5 = DownSample(512,512)
    self.down6 = DownSample(512,512)
    self.down7 = DownSample(512,512)
    self.down8 = DownSample(512,512)

    self.up1 = UpSample(512,512)
    self.up2 = UpSample(1024, 512)
    self.up3 = UpSample(1024, 512)
    self.up4 = UpSample(1024, 512)
    self.up5 = UpSample(1024, 256)
    self.up6 = UpSample(512, 128)
    self.up7 = UpSample(256, 64)

    self.final = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.ZeroPad2d((1,0,1,0)),
      nn.Conv2d(128,3,4,padding=1), # out_channels
      nn.Tanh(),
    )

  def forward(self, x):
    # U-Net generator with skip connections from encoder to decoder

    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    d4 = self.down4(d3)
    d5 = self.down5(d4)
    d6 = self.down6(d5)
    d7 = self.down7(d6)
    d8 = self.down8(d7)
    u1 = self.up1(d8, d7)
    u2 = self.up2(u1, d6)
    u3 = self.up3(u2, d5)
    u4 = self.up4(u3, d4)
    u5 = self.up5(u4, d3)
    u6 = self.up6(u5, d2)
    u7 = self.up7(u6, d1)
    u8 = self.final(u7)

    return u8
  

class Pix2PixDiscriminator(nn.Module):
  def __init__(self, in_channels=6,):
    super(Pix2PixDiscriminator, self).__init__()

    self.model = nn.Sequential(
      nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2,inplace=True),

      nn.Conv2d(128, 256, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),

      nn.ZeroPad2d((1, 0, 1, 0)),
      nn.Conv2d(256, 1, 4, padding=1, bias=False),
    )

  def forward(self, img_A, img_B):
    #  Here we concatenate the images on their channels
    img_input = torch.cat((img_A, img_B), 1)
    return self.model(img_input)
