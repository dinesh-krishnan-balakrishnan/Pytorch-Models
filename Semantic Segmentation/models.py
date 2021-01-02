import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
          Block(3, 16),
          Block(16, 32, stride = 2),
          Block(32, 64, stride = 2),
          Block(64, 128, stride = 2),
          torch.nn.Flatten(),
          torch.nn.Linear(128 * 8 * 8, 128),
          torch.nn.Dropout(0.4),
          torch.nn.PReLU(),
          torch.nn.Linear(128, 6)
        )

    def forward(self, x):
        x = (x / torch.max(x)) * 2 - 1
        return self.network(x)

class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.start = torch.nn.BatchNorm2d(3)
        self.down1 = FCNBlock(3, 32, stride = 2)
        self.down2 = FCNBlock(32, 64, stride = 2)
        self.down3 = FCNBlock(64, 128, stride = 2)
        self.down4 = FCNBlock(128, 256, stride = 2)
        self.up1 = InverseBlock(256, 128, stride = 2)
        self.up2 = InverseBlock(128, 64, stride = 2)
        self.up3 = InverseBlock(64, 32, stride = 2)
        self.up4 = InverseBlock(32, 16, stride = 2)
        self.linear = torch.nn.Conv2d(16, 5, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        H = x.size(2)
        W = x.size(3)
        x = (x / torch.max(x)) * 2 - 1

        x = self.start(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.linear(x)

        x = x[:, :, :H, :W]
        return x

class Block(torch.nn.Module):
  def __init__(self, _input, _output, stride = 1):
    super().__init__()
    self.network = torch.nn.Sequential(
      torch.nn.Conv2d(_input, _output, kernel_size = 3, padding = 1, stride = stride, bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.PReLU(),
      torch.nn.Conv2d(_output, _output, kernel_size = 3, padding = 1, bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.PReLU()
    )
    self.downsample = torch.nn.Sequential(
      torch.nn.Conv2d(_input, _output, kernel_size = 1, stride = stride, bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.PReLU()
    )

  def forward(self, x):
    return self.network(x) + self.downsample(x)

class FCNBlock(torch.nn.Module):
  def __init__(self, _input, _output, stride = 1):
    super().__init__()
    self.network = torch.nn.Sequential(
      torch.nn.Conv2d(_input, _output, kernel_size = 3, padding = 1, stride = stride, bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.Dropout(0.2),
      torch.nn.PReLU(),
      torch.nn.Conv2d(_output, _output, kernel_size = 3, padding = 1, bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.PReLU()
    )
    self.downsample = torch.nn.Sequential(
      torch.nn.Conv2d(_input, _output, kernel_size = 1, stride = stride, bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.PReLU()
    )

  def forward(self, x):
    return self.network(x) + self.downsample(x)

class InverseBlock(torch.nn.Module):
  def __init__(self, _input, _output, stride = 1):
    super().__init__()
    self.network = torch.nn.Sequential(
      torch.nn.ConvTranspose2d(_input, _output, kernel_size = 4, padding = 1, stride = stride, bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.Dropout(0.2),
      torch.nn.PReLU(),
      torch.nn.Conv2d(_output, _output, kernel_size = 3, padding = 1, bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.PReLU()
    )
    self.upsample = torch.nn.Sequential(
      torch.nn.ConvTranspose2d(_input, _output, kernel_size = 1, stride = stride, bias = False, output_padding = 1),
      torch.nn.BatchNorm2d(_output),
      torch.nn.PReLU()
    )

  def forward(self, x):
    return self.network(x) + self.upsample(x)

model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
