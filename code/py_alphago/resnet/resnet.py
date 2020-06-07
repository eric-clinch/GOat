import torch.nn as nn
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, inChannels, outChannels, downsample, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels,
                               3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels,
                               3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannels)

        self.downsample = downsample
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        return self.relu(x)


class Resnet(nn.Module):
    def __init__(self, inChannels, board_size):
        super(Resnet, self).__init__()
        self.inChannels = inChannels

        self.conv = self.makeConvLayer(32)
        self.res1 = self.makeResidualBlock(32)
        self.res2 = self.makeResidualBlock(32)
        self.res3 = self.makeResidualBlock(32)
        self.res4 = self.makeResidualBlock(32)
        self.valueLayer = nn.Linear(
            board_size * board_size * self.inChannels, 1)
        self.policyLayer = nn.Linear(
            board_size * board_size * self.inChannels, board_size * board_size + 1)
        self.soft_max = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def Load(self, file_path):
        self.load_state_dict(torch.load(
            file_path, map_location=torch.device(DEVICE)))

    def makeConvLayer(self, outChannels):
        result = ConvBlock(self.inChannels, outChannels)
        self.inChannels = outChannels
        return result

    def makeResidualBlock(self, outChannels, stride=1):
        downsample = None
        if self.inChannels != outChannels or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inChannels, outChannels,
                          kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(outChannels)
            )
        result = ResidualBlock(
            self.inChannels, outChannels, downsample, stride)
        self.inChannels = outChannels
        return result

    def forward(self, x):
        if len(x.shape) < 4:
            # This is a singleton datapoint. Pack it into a batch of size 1
            x = x.unsqueeze(0)

        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = x.view(x.size(0), -1)

        value = self.sigmoid(self.valueLayer(x))
        policy = self.soft_max(self.policyLayer(x))

        return value, policy
