import numpy as np
import torch
from torch import nn
import sys
import torch.nn.functional as F


class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w*h).permute(0, 2, 1)
        pooled = F.max_pool1d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)

#sys.path.append('/Users/aniruddha/Google Drive/JHU_courses/Deep_Learning/Project Presentation/maskrcnn-benchmark-master')
#from maskrcnn_benchmark.layers import roi_pool

def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
      nn.ConvTranspose2d(ch_coarse, ch_fine, kernel_size = 4, stride = 2, padding = 1, bias = True),
      nn.ReLU()
  )

class PanNet(nn.Module):
    def __init__(self):
        super(PanNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = upsample(64, 64)
        self.conv6 = upsample(128, 64)
        self.conv7 = upsample(128, 64)

        self.conv8_init = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        #self.conv8_final = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.linear8 = nn.Sequential(nn.Linear(in_features = 1*128*128, out_features = 1024), nn.BatchNorm2d(1024))
        self.linear9 = nn.Sequential(nn.Linear(in_features = 1*64*64, out_features = 512), nn.BatchNorm2d(512))
        self.linear10 = nn.Sequential(nn.Linear(in_features = 1*32*32, out_features = 256), nn.BatchNorm2d(256))

        self.channel_pool256 = ChannelPool(256)
        self.channel_pool192 = ChannelPool(192)

        self.fc1 = nn.Sequential(nn.Linear(in_features = 128*128 + 64*64 + 32*32, out_features = 1024))
        self.fc2 = nn.Sequential(nn.Linear(in_features = 1024, out_features = 32))
        self.fc3 = nn.Sequential(nn.Linear(in_features = 32, out_features = 4))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #FPN
        out1 = self.conv1(x)
        out1 = self.batch_norm64(out1)
        out1 = self.max_pool(out1) #256
        out1= self.relu(out1)

        out2 = self.conv2 (out1)
        out2 = self.batch_norm64(out2)
        out2  = self.relu(out2)
        out2 = self.max_pool(out2) #128

        out3 = self.conv3(out2)
        out3 = self.batch_norm64(out3)
        out3 = self.relu(out3)
        out3 = self.max_pool(out3) #64

        out4 = self.conv4(out3) 
        out4 = self.batch_norm64(out4)
        out4 = self.relu(out4)
        out4 = self.max_pool(out4) #32

        out5 = self.conv5(out4)
        out5 = self.batch_norm64(out5) 
        out5 = self.relu(out5)
        out5 = torch.cat((out3, out5), dim = 1)  # 64

        out6 = self.conv6(out5) 
        out6 = self.batch_norm64(out6)
        out6 = self.relu(out6)
        out6 = torch.cat((out2, out6), dim = 1) #128

        out7 = self.conv7(out6)
        out7 = self.batch_norm64(out7)
        out7 = self.relu(out7)
        out7 = torch.cat((out1, out7), dim = 1) #256

        #Bottom up path augmentation
        out8 = self.conv8_init(out7)
        out8 = self.batch_norm128(out8)
        out8 = self.relu(out8)
        out8 = self.max_pool(out8)
        out8 = torch.cat((out6, out8), dim = 1) #128
        out8_max = self.channel_pool256(out8)
        out8_max = out8_max.view(-1, 1, 128*128)

        out9 = self.conv8(out8)
        out9 = self.batch_norm128(out9)
        out9 = self.relu(out9)
        out9 = self.max_pool(out9)
        out9 = torch.cat((out5, out9), dim = 1) #64
        out9_max = self.channel_pool256(out9)
        out9_max = out9_max.view(-1, 1, 64*64)

        out10 = self.conv8(out9)
        out10 = self.batch_norm128(out10)
        out10 = self.relu(out10)
        out10 = self.max_pool(out10)
        out10 = torch.cat((out4, out10), dim = 1) #32
        out10_max = self.channel_pool192(out10)
        out10_max = out10_max.view(-1, 1, 32*32)
        
        # Adaptive pooling
        out_final = torch.cat((out8_max, out9_max, out10_max), dim = 2)
        
        #Box regression network
        out_final = self.fc1(out_final)
        out_final = self.fc2(out_final)
        out_final = self.fc3(out_final)

        #out_final = 511*self.sigmoid(out_final)

        return(out_final)

class autoencoder_improved(nn.Module):
    def __init__(self):
        super(autoencoder_improved, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        # make sure you keep this layer during your autoencoder training, this
        # will be used for Q3-(c) fully connected layer
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # IMPORTANT: in Q3-(c), please delete the above decoder layer, and use
        # the linear layer to build fully-connection layers.
        return (x)

class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = upsample(64, 64)
        self.conv6 = upsample(128, 64)
        self.conv7 = upsample(128, 64)

        self.auto_conv = upsample(128, 3)

        self.conv8_init = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        #self.conv8_final = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.linear8 = nn.Sequential(nn.Linear(in_features = 1*128*128, out_features = 1024), nn.BatchNorm2d(1024))
        self.linear9 = nn.Sequential(nn.Linear(in_features = 1*64*64, out_features = 512), nn.BatchNorm2d(512))
        self.linear10 = nn.Sequential(nn.Linear(in_features = 1*32*32, out_features = 256), nn.BatchNorm2d(256))

        self.channel_pool256 = ChannelPool(256)
        self.channel_pool192 = ChannelPool(192)

        self.fc1 = nn.Sequential(nn.Linear(in_features = 128*128 + 64*64 + 32*32, out_features = 1024))
        self.fc2 = nn.Sequential(nn.Linear(in_features = 1024, out_features = 32))
        self.fc3 = nn.Sequential(nn.Linear(in_features = 32, out_features = 4))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #Autoencoder
        out1 = self.conv1(x)
        out1 = self.batch_norm64(out1)
        out1 = self.max_pool(out1) #256
        out1= self.relu(out1)

        out2 = self.conv2 (out1)
        out2 = self.batch_norm64(out2)
        out2  = self.relu(out2)
        out2 = self.max_pool(out2) #128

        out3 = self.conv3(out2)
        out3 = self.batch_norm64(out3)
        out3 = self.relu(out3)
        out3 = self.max_pool(out3) #64

        out4 = self.conv4(out3) 
        out4 = self.batch_norm64(out4)
        out4 = self.relu(out4)
        out4 = self.max_pool(out4) #32

        out5 = self.conv5(out4)
        out5 = self.batch_norm64(out5) 
        out5 = self.relu(out5)
        out5 = torch.cat((out3, out5), dim = 1)  # 64

        out6 = self.conv6(out5) 
        out6 = self.batch_norm64(out6)
        out6 = self.relu(out6)
        out6 = torch.cat((out2, out6), dim = 1) #128

        out7 = self.conv7(out6)
        out7 = self.batch_norm64(out7)
        out7 = self.relu(out7)
        out7 = torch.cat((out1, out7), dim = 1) #256

        out_auto = self.auto_conv(out7)

        #Bottom up path augmentation
        out8 = self.conv8_init(out7)
        out8 = self.batch_norm128(out8)
        out8 = self.relu(out8)
        out8 = self.max_pool(out8)
        out8 = torch.cat((out6, out8), dim = 1) #128
        out8_max = self.channel_pool256(out8)
        out8_max = out8_max.view(-1, 1, 128*128)

        out9 = self.conv8(out8)
        out9 = self.batch_norm128(out9)
        out9 = self.relu(out9)
        out9 = self.max_pool(out9)
        out9 = torch.cat((out5, out9), dim = 1) #64
        out9_max = self.channel_pool256(out9)
        out9_max = out9_max.view(-1, 1, 64*64)

        out10 = self.conv8(out9)
        out10 = self.batch_norm128(out10)
        out10 = self.relu(out10)
        out10 = self.max_pool(out10)
        out10 = torch.cat((out4, out10), dim = 1) #32
        out10_max = self.channel_pool192(out10)
        out10_max = out10_max.view(-1, 1, 32*32)
        
        # Adaptive pooling
        out_final = torch.cat((out8_max, out9_max, out10_max), dim = 2)
        
        #Box regression network
        out_final = self.fc1(out_final)
        out_final = self.fc2(out_final)
        out_final = self.fc3(out_final)

        #out_final = 511*self.sigmoid(out_final)

        return((out_final, out_auto))

class predictClass(nn.Module):
    def __init__(self):
        super(predictClass, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, padding=0), nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, padding=0), nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(8), nn.ReLU())

        self.classifier = nn.Sequential(nn.Linear(in_features = 128*128*8, out_features = 1024), nn.Linear(in_features = 1024, out_features = 16), nn.Linear(in_features = 16, out_features = 8))

        self.softmax = nn.Softmax(dim = 2)
    
    def forward(self, x):
        out = self.encoder(x)
        out = out.view(-1, 1, 8*128*128)
        out = self.classifier(out)
        out = self.softmax(out)
        return(out)


class IOULoss(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(IOULoss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.relu = nn.ReLU()
    
    def forward(self, bbox, bbox_pred):
        area1 = (bbox[:, 0, 2] - bbox[:, 0, 0])*(bbox[:, 0, 3] - bbox[:, 0, 1])
        area2 = (bbox_pred[:, 0, 2] - bbox_pred[:, 0, 0]) * \
            (bbox_pred[:, 0, 3] - bbox_pred[:, 0, 1])
        area_intersection = (torch.min(bbox[:, 0, 2], bbox_pred[:, 0, 2]) - torch.max(bbox[:, 0, 0], bbox_pred[:, 0, 0]))*(
            torch.min(bbox[:, 0, 3], bbox_pred[:, 0, 3]) - torch.max(bbox[:, 0, 1], bbox_pred[:, 0, 1]))

        loss = (area_intersection + 1e-4)/(area1 + area2 - area_intersection + 1e-4)
        loss = self.relu(loss)
        loss = torch.mean(loss, dim = 0)
        return(loss)




if __name__ == '__main__':
    #random_img = torch.rand(( 1, 3, 512, 512))
    #encodernet = EncoderNet()
    #predicter = predictClass()
    #output = encodernet(random_img)
    #output = predicter(random_img)
    #print(output)

    bbox_pred = torch.tensor(
        [[368, 269, 429, 340], [368, 269, 429, 340]], dtype=torch.float32)
    bbox = torch.tensor(
        [[302, 274, 319, 295], [302, 274, 319, 295]], dtype=torch.float32)

    bbox_pred = bbox_pred.view(-1, 1, 4)
    bbox = bbox.view(-1, 1, 4)
    loss = IOULoss()
    
    print(loss(bbox = bbox, bbox_pred= bbox_pred))

