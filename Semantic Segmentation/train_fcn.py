import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, DENSE_LABEL_NAMES
from . import dense_transforms
import torch.utils.tensorboard as tb

# Student Declarations
TRAINING_DATA_PATH = 'dense_data/train'
TEST_DATA_PATH = 'dense_data/valid'
BATCH_SIZE = 32
EPOCHS = 20


def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    from os import path
    model = FCN().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Loading the training and testing data.
    training_data = load_dense_data(TRAINING_DATA_PATH, batch_size = BATCH_SIZE)
    testing_data = load_dense_data(TEST_DATA_PATH, batch_size = BATCH_SIZE)
    val_data, val_labels = [
      [data, labels] for data, labels in load_dense_data(TEST_DATA_PATH, batch_size = 1)
    ][0]
    val_data, val_labels = val_data.to(device), val_labels.to(device)

    # Optimizer & Loss 
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-1)
    DENSE_WEIGHTS = (1.0 / torch.FloatTensor(DENSE_CLASS_DISTRIBUTION)).to(device)
    loss_func = torch.nn.CrossEntropyLoss(DENSE_WEIGHTS)

    converter = dense_transforms.ToTensor()
    result_tracker = ConfusionMatrix()

    for epoch in range(EPOCHS):
        model.train()
        print(epoch)
        # Iterates through the the batched data.
        for data, labels in training_data:
          # Adds the batch to the GPU
          data = data.to(device)
          labels = labels.long().to(device)

          # Determines loss based on the results of the model.
          results = model(data)
          loss = loss_func(results, labels)

          # Updates the parameters based on the loss and gradients.
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        # Logging Results
        model.eval()
        results = model(val_data)
        result_tracker.add(results.argmax(1), val_labels)
        print(result_tracker.iou, result_tracker.global_accuracy)

    save_model(model)



def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
