import numpy as np
from os import path

import torch
from torchvision import transforms

class FCN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.down1 = Block(3, 16)
    self.down2 = Block(16, 32)
    self.down3 = Block(32, 64)
    self.down4 = Block(64, 128)
    self.up1 = InverseBlock(128, 64)
    self.up2 = InverseBlock(64, 32)
    self.up3 = InverseBlock(32, 16)
    self.up4 = InverseBlock(16, 3)
    self.linear = torch.nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    H = x.shape[2]
    W = x.shape[3]

    x = self.down4(self.down3(self.down2(self.down1(x))))
    x = self.up4(self.up3(self.up2(self.up1(x))))
    x = self.linear(x)
    x = x[:, :, :H, :W]
    return x

class Block(torch.nn.Module):
  def __init__(self, _input, _output, K = 3, stride = 2):
    super().__init__()
    self.network = torch.nn.Sequential(
      torch.nn.Conv2d(_input, _output, kernel_size = K, padding = (K // 2), stride = stride, bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.PReLU(),
      torch.nn.Conv2d(_output, _output, kernel_size = K, padding = (K // 2), bias = False),
      torch.nn.BatchNorm2d(_output),
      torch.nn.PReLU(),
      torch.nn.Conv2d(_output, _output, kernel_size = K, padding = (K // 2), bias = False),
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
  def __init__(self, _input, _output, K = 3, stride = 2):
    super().__init__()
    self.network = torch.nn.Sequential(
      torch.nn.ConvTranspose2d(_input, _output, kernel_size = K, padding = (K // 2), stride = stride, output_padding = 1),
      torch.nn.PReLU()
    )

  def forward(self, x):
    return self.network(x)

# ---------------------------------------------

class HockeyPlayer:
    def __init__(self, player_id=0):
      # Generated State Constants
      self.player_id = player_id
      self.team = player_id % 2

      # Hardcoded Kart Type & Locations
      self.kart = 'tux'
      self.goal = 64.5 * (1 if (self.team == 0) else -1)
      self.position = 3 if (self.player_id - self.team == 0) else -3

      # Model Constants
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.resize = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize((150, 200)),
          transforms.ToTensor()
      ])
      self.model = FCN().to(self.device)
      self.model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'FCN_100.th')))

      # Reset Constants
      self.resetting = False
      self.has_reset = False
      self.zeros = 0
      self.previous = None
      self.backing = 0

    def act(self, image, player_info):
      # Value Generation
      front = np.float32(player_info.kart.front)[[0, 2]]
      kart = np.float32(player_info.kart.location)[[0, 2]]
      x, y, range, exists = self.get_coords(image)

      # State Check
      self.check_reset(kart)
      if exists:
        self.resetting = False
      if self.distance(kart, [self.position, -self.goal]) < 5:
        self.resetting = False
        self.zeros = 0
      
      if (abs(x) > 75) and y > 85:
        self.backing = 8
      
      if self.backing > 0:
        self.backing -= 1
        return {
          'steer': -np.sign(x) * (1.5 if self.backing < 4 else 0),
          'acceleration': 0,
          'brake': True,
          'drift': False,
          'nitro': not self.has_reset, 'rescue': False
        }

      if not (self.resetting):
        # Update Action & Reset State
        if not exists:
          self.zeros += 1
          if self.zeros >= (40 if not self.has_reset else 25):
            self.resetting = self.has_reset = True
          steer_vec = self.get_steer_vec(front, kart, [self.position * 8, -self.goal / 2]) / 2 if (self.has_reset) else 0

        else:
          self.zeros = max(self.zeros - 2, 0)
          y /= 30.
          goal_steer_vec = self.get_steer_vec(front, kart, [0, self.goal])
          if abs(goal_steer_vec) > 0.25:
            modifier = (np.sign(goal_steer_vec) * range / 4)
            steer_vec = (np.sign(x + modifier) * y) + (1.1 * goal_steer_vec)     
          else:
            steer_vec = (np.sign(x) * y)

        return {
          'steer': steer_vec,
          'acceleration': 1 if not self.has_reset else 0.5,
          'brake': False,
          'drift': False,
          'nitro': not self.has_reset, 'rescue': False
        }

      return {
          'steer': -self.get_steer_vec(front, kart, [self.position, -self.goal]),
          'acceleration': 0,
          'brake': True,
          'drift': False,
          'nitro': False, 'rescue': False
      }

    def check_reset(self, kart):
      if self.distance(kart, self.previous) > 4:
        self.zeros = 0
        self.has_reset = False
        self.resetting = False
        self.backing = 0

      self.previous = kart

    # Getting Puck Coordinates
    def get_coords(self, image):
        # Passing the image through the model.
        image = self.resize(image).to(self.device)
        result = self.model(image.reshape(1, 3, 150, 200))[0]
        # Removing all non-positive results from the tensor.
        result[result < 0] = 0
        nz = torch.nonzero(result, as_tuple = True)
        # Calculating & Returning Results
        if nz[0].numel() > 0:
          return (
              nz[2].float().mean().item() - 100, 
              nz[1].float().mean().item(),
              nz[2].float().max().item() - nz[2].float().min().item(),
              True 
          )
        return 0, 0, 0, False


    # Calculates the steering vector to reached a global point on the map.
    def get_steer_vec(self, front, kart, location):
        # Calculating Kart & Desired Position Vector
        kart_vec = front - kart
        kart_vec /= np.linalg.norm(kart_vec)
        v = location - kart
        v /= np.linalg.norm(v)
        # Cross-Product
        return np.sign(np.cross(kart_vec, v)) * np.arccos(np.dot(kart_vec, v) + 1e-8)

    # Euclidean Distance
    def distance(self, kart, location):
        if location is None:
          return 0
        return np.sqrt(np.square(kart[0] - location[0]) + np.square(kart[1] - location[1]))