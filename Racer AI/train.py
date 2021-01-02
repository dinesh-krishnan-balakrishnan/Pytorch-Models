from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

# Student Declarations
DATA_PATH = 'drive_data'
BATCH_SIZE = 256
EPOCHS = 100

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    from os import path
    model = Planner().to(device)
    train_logger = tb.SummaryWriter('logs', flush_secs=1)

    # Loading training data.
    training_data = load_data(DATA_PATH, batch_size = BATCH_SIZE)

    # Optimizer & Loss 
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-7)

    # Algorithm
    for epoch in range(EPOCHS):
      print(epoch)
      for data, labels in training_data:
        # Performing Prediction
        data = data.to(device)
        labels = labels.to(device)
        results = model(data)

        # Calculating Loss
        loss = loss_func(results, labels)

        # Updating Weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 25 == 0:
            save_model(model, f'planner{epoch}.th')

    return 

    # Logging Results
    model.eval()
    steps = 0
    for data, labels in training_data:
      data = data.to(device)
      labels = labels.to(device)
      results = model(data)      

      for index in range(len(results)):
        steps += 1
        log(train_logger, data[index], labels[index], results[index], steps)
        

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz_5', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()

    train()
