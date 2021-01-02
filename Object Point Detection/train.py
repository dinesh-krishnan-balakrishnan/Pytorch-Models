import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb

# Student Declarations
TRAINING_DATA_PATH = 'dense_data/train'
TEST_DATA_PATH = 'dense_data/valid'
BATCH_SIZE = 32
EPOCHS = 150

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    from os import path
    model = Detector().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Loading training & testing data.
    training_data = load_detection_data(TRAINING_DATA_PATH, batch_size = BATCH_SIZE)
    val_data, val_labels = [
      [data, labels] for data, labels, extra in load_detection_data(TEST_DATA_PATH, batch_size = 16)
    ][0]
    val_data, val_labels = val_data.to(device), val_labels.to(device)

    # Optimizer & Loss 
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4, weight_decay = 1e-6)
    loss_func = torch.nn.BCEWithLogitsLoss(reduction = 'none')

    # Logger
    logger = tb.SummaryWriter('/logs/train/1')

    # Algorithm
    for epoch in range(EPOCHS):
      print(epoch)
      model.train()
      for data, labels, extra in training_data:
        # Performing Prediction
        data = data.to(device)
        labels = labels.to(device)
        results = model(data)

        # Calculating Loss
        BCE = loss_func(results, labels)
        Pt = torch.exp(-BCE)
        focal_loss = ((1 - Pt) ** 2 * BCE).mean()

        # Updating Weights
        optimizer.zero_grad()
        focal_loss.backward()
        optimizer.step()

      # Logging Results
      model.eval()
      results = model(val_data)
      log(logger, val_data, val_labels, results, epoch)

      if epoch == 100:
        save_model(model, name = 'det1.th')

      if epoch == 125:
        save_model(model, name = 'det2.th')


    save_model(model) 

def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
