import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.zeros(batch_size, residual_channel - shortcut_channel, 
                          featuremap_size[0], featuremap_size[1]).to(out.device))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 2:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=8, stride=stride, padding=3, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.zeros(batch_size, residual_channel - shortcut_channel, 
                          featuremap_size[0], featuremap_size[1]).to(out.device))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut
        return out

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Average pooling
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        # Channel attention
        y_avg = self.fc(y_avg).view(b, c, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1)
        y = self.sigmoid(y_avg + y_max)
        return x * y.expand_as(x)

class PResNet(nn.Module):
    def __init__(self, depth, alpha, num_classes, n_bands, avgpoosize, inplanes, bottleneck=True):
        super(PResNet, self).__init__()

        if bottleneck:
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            n = (depth - 2) // 6
            block = BasicBlock

        self.addrate = alpha / (3 * n * 1.0)
        self.inplanes = inplanes

        # 3D Convolution layers
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(5, 3, 3), stride=1, padding=(0,1,1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(0,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        # Add CCA layer after 3D convolutions
        self.cca = CCALayer(16 * (32-6))  # Channel dimension after flattening

        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d((32-6)*16, self.input_featuremap_dim, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        # Pyramid layers with SE blocks
        self.featuremap_dim = self.input_featuremap_dim
        self.layer1 = self.pyramidal_make_layer(block, n)
        self.se1 = SELayer(self.input_featuremap_dim)

        self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
        self.se2 = SELayer(self.input_featuremap_dim)

        self.layer3 = self.pyramidal_make_layer(block, n, stride=2)
        self.se3 = SELayer(self.input_featuremap_dim)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Extra Fully Connected Layer
        self.fc1 = nn.Linear(self.final_featuremap_dim, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)

        self._initialize_weights()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2))

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), 
                          stride, downsample))

        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(
                block(int(round(self.featuremap_dim)) * block.outchannel_ratio,
                      int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim = temp_featuremap_dim

        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Initial 3D processing
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)

        # Reshape and apply CCA
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3), x.size(4))
        x = self.cca(x)  # Apply cross-channel attention

        # Convert to 2D and process through pyramid network
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.se1(x)

        x = self.layer2(x)
        x = self.se2(x)

        x = self.layer3(x)
        x = self.se3(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Extra Fully Connected Layer
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)

        return x
