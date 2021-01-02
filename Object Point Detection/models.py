import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """

    peaks, heatmap_indices = F.max_pool2d(
      heatmap[None, None], 
      kernel_size = max_pool_ks, 
      padding = max_pool_ks // 2, 
      stride = 1,
      return_indices = True
    )
    peaks = torch.flatten(peaks[0, 0])
    heatmap_indices = torch.flatten(heatmap_indices[0, 0])

    peak_condition = (peaks == torch.flatten(heatmap))
    peaks = peaks[peak_condition]
    heatmap_indices = heatmap_indices[peak_condition]

    max_peaks, peak_indices = torch.topk(peaks, min(max_det, len(peaks)))
    max_indices = max_peaks > min_score

    valid_indices = heatmap_indices[peak_indices[max_indices]] 
    H = valid_indices // heatmap.shape[1]
    W = valid_indices % heatmap.shape[1]

    return [(heatmap[h, w], w, h) for h, w in zip(H, W)]


class Detector(torch.nn.Module):
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

          self.DownSample = torch.nn.Conv2d(_input, _output, kernel_size = 1, stride = stride)

      def forward(self, x):
          I1 = F.relu(self.BN1(self.C1(x)))
          I2 = F.relu(self.BN2(self.C2(I1)))
          I3 = F.relu(self.BN3(self.C3(I2)))
          return F.relu(I3 + self.DownSample(x))

    class UpConv(torch.nn.Module):
        def __init__(self, _input, _output, K_size = 3, stride = 2):
            super().__init__()
            self.CT = torch.nn.ConvTranspose2d(_input, _output, kernel_size = K_size, 
                        padding = (K_size // 2), stride = stride, output_padding = 1)
        
        def forward(self, x):
          return F.relu(self.CT(x))

    def __init__(self):
        super().__init__()

        self.D1 = self.DownConv(3, 32)
        self.D2 = self.DownConv(32, 64)
        self.D3 = self.DownConv(64, 128)
        self.D4 = self.DownConv(128, 256)
        self.U1 = self.UpConv(256, 128)
        self.U2 = self.UpConv(128, 64)
        self.U3 = self.UpConv(64, 32)
        self.U4 = self.UpConv(32, 16)
        self.Linear = torch.nn.Conv2d(16, 3, kernel_size = 1, stride = 1, padding = 0)


    def forward(self, x):
        H = x.size(2)
        W = x.size(3)
        x = (x / torch.max(x)) * 2 - 1 

        x = self.D4(self.D3(self.D2(self.D1(x))))
        x = self.U4(self.U3(self.U2(self.U1(x))))
        x = self.Linear(x)
        return x

    def detect(self, image):
        prediction = self.forward(image[None])[0]
        return [
          [(score, x, y, 7.0, 7.0) for (score, x, y) in extract_peak(_class, max_det = 30)] 
          for _class in prediction
        ]


def save_model(model, name = 'det.th'):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
