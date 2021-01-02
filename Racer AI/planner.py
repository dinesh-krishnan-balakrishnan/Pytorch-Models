import torch
import torch.nn.functional as F

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    class DownConv(torch.nn.Module):
      def __init__(self, _input, _output, K_size = 3, stride = 2):
          super().__init__()
          self.C1 = torch.nn.Conv2d(_input, _output, kernel_size = K_size, 
                      padding = (K_size // 2), stride = stride, bias = False)
          self.C2 = torch.nn.Conv2d(_output, _output, kernel_size = K_size, 
                      padding = (K_size // 2), bias = False)
          self.C3 = torch.nn.Conv2d(_output, _output, kernel_size = K_size, 
                      padding = (K_size // 2), bias = False)

          self.BN1 = torch.nn.BatchNorm2d(_output)
          self.BN2 = torch.nn.BatchNorm2d(_output)
          self.BN3 = torch.nn.BatchNorm2d(_output)

          self.R1 = torch.nn.PReLU()
          self.R2 = torch.nn.PReLU()
          self.R3 = torch.nn.PReLU()
          self.R4 = torch.nn.PReLU()

          self.DownSample = torch.nn.Conv2d(_input, _output, kernel_size = 1, stride = stride)

      def forward(self, x):
          I1 = self.R1(self.BN1(self.C1(x)))
          I2 = self.R2(self.BN2(self.C2(I1)))
          I3 = self.R3(self.BN3(self.C3(I2)))
          return self.R4(I3 + self.DownSample(x))

    class UpConv(torch.nn.Module):
        def __init__(self, _input, _output, K_size = 3, stride = 2):
            super().__init__()
            self.CT = torch.nn.ConvTranspose2d(_input, _output, kernel_size = K_size, 
                        padding = (K_size // 2), stride = stride, output_padding = 1)
            self.R = torch.nn.PReLU()

        def forward(self, x):
          return self.R(self.CT(x))

    def __init__(self):
        super().__init__()

        self.D1 = self.DownConv(3, 32)
        self.D2 = self.DownConv(32, 64)
        self.D3 = self.DownConv(64, 128)
        self.U1 = self.UpConv(128, 64)
        self.U2 = self.UpConv(64, 32)
        self.U3 = self.UpConv(32, 16)
        self.Linear = torch.nn.Conv2d(16, 1, kernel_size = 1, stride = 1, padding = 0)


    def forward(self, x):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = (x / 255) * 2 - 1 

        x = self.D3(self.D2(self.D1(x)))
        x = self.U3(self.U2(self.U1(x)))
        x = self.Linear(x)
        return spatial_argmax(x.squeeze(dim = 1))


def save_model(model, name):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


def test_planner(pytux, track, verbose=False):
    from .controller import control

    track = [track] if isinstance(track, str) else track
    planner = load_model().eval()

    for t in track:
        steps, how_far = pytux.rollout(t, control, planner, max_frames=1000, verbose=verbose)
        print(steps, how_far)


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')

    pytux = PyTux()
    test_planner(pytux, **vars(parser.parse_args()))
    pytux.close()