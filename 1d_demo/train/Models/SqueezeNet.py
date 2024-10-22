import torch

class FireModule(torch.nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand1x3_channels):
        super(FireModule, self).__init__()
        self.squeeze = torch.nn.Conv1d(in_channels, squeeze_channels, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.expand1x1 = torch.nn.Conv1d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x3 = torch.nn.Conv1d(squeeze_channels, expand1x3_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.squeeze(x)
        x = self.relu(x)
        out1x1 = self.expand1x1(x)
        out1x3 = self.expand1x3(x)
        out = torch.cat([out1x1, out1x3], dim=1)
        return self.relu(out)

class SqueezeNet(torch.nn.Module):
    def __init__(self,in_channels=3, classes=10):
        super(SqueezeNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 96, kernel_size=7, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),
            FireModule(512, 64, 256, 256)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv1d(512, classes, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool1d((1))
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x

if __name__ == "__main__":
    model = SqueezeNet(in_channels=3,classes=10)
    input = torch.randn(1,3,224)
    output = model(input)
    print(output.shape)
