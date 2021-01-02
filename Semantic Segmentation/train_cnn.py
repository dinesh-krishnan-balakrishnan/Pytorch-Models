from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb

EPOCHS = 20
BATCH_SIZE = 128
TRAINING_DATA_PATH = 'data/train'
TEST_DATA_PATH = 'data/valid'

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    from os import path
    model = CNNClassifier().to(device)
    model.train()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Loading the training data.
    training_data = load_data(TRAINING_DATA_PATH, batch_size = BATCH_SIZE)
    testing_data = load_data(TEST_DATA_PATH, batch_size = BATCH_SIZE)

    # Optimizer & Loss 
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-2)
    loss_func = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
      model.train()
      print(epoch)
      # Iterates through the the batched data.
      for data, labels in training_data:
        # Adds the batch to the GPU
        data = data.to(device)
        labels = labels.to(device)

        # Determines loss based on the results of the model.
        results = model(data)
        loss = loss_func(results, labels)

        # Updates the parameters based on the loss and gradients.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


      model.eval()
      val_results = []
      for data, labels in testing_data:
        data = data.to(device)
        labels = labels.to(device)
        temp = torch.argmax(model(data), dim = 1)
        val_results.append((temp == labels).float().mean())

      print(torch.FloatTensor(val_results).to(device).mean())


    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
