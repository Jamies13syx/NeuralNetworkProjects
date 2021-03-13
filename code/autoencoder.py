import time, os, datetime
import argparse
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
learning_rate = 0.001
global nEpochs
global log_file
nEpochs = 1

dropout = 0.0
L2_lambda = 0.0
mean = 0.0
std = 1.0
test_batch_size = 1000
log_interval = 100
# hyperparameters set


def display_image(img):
    npimg = img * std + mean
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def log_message(outf, message):
    print(message)
    if not outf is None:
        outf.write(message)
        outf.write("\n")
        outf.flush()


class Autoencoder(nn.Module):
    def __init__(self, specs):
        super(Autoencoder, self).__init__()
        C0, C1, C2, C3, kernel_size, padding = specs

        self.encoder = nn.Sequential(
            nn.Conv2d(C0, C1, kernel_size, padding=padding, stride=2),  # [batch, 12, 16, 16]
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(C1, C2, kernel_size, padding=padding, stride=2),  # [batch, 24, 8, 8]
            nn.LeakyReLU(negative_slope=0.01),
            # nn.Conv2d(C2, C3, kernel_size, padding=padding, stride=2),  # [batch, 48, 4, 4]
            # nn.LeakyReLU(negative_slope=0.01),

        )  # encode to  H:4  W:4  Chanel: 48 tensor
        self.decoder = nn.Sequential(

            # nn.ConvTranspose2d(C3, C2, kernel_size, padding=padding, stride=2),  # [batch, 24, 8, 8]
            # nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(C2, C1, kernel_size, padding=padding, stride=2),  # [batch, 12, 16, 16]
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(C1, C0, kernel_size, padding=padding, stride=2),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )  # decode to  H:32  W:32  Chanel: 3 tensor

    def forward(self, x):
        encoded = self.encoder(x)  # encode to  H:4  W:4  Chanel: 48 tensor, 3 layers
        decoded = self.decoder(encoded) # decode to  H:32  W:32  Chanel: 3 layers
        return decoded


def train(model, train_loader, test_loader):
    # define the loss function
    criterion = nn.BCELoss(reduction="sum")

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

    start = time.time()
    w_decay = 0.8
    for e in range(nEpochs):
        total_train_images = 0
        total_train_loss = 0
        train_images = 0
        train_loss = 0
        w_images = 0
        w_loss = 0

        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):

            optimizer.zero_grad()
            output = model(Variable(data))

            loss = criterion(output, Variable(data))  # target is data its self
            loss.backward()
            optimizer.step()
            train_images += len(data)
            train_loss += loss.data.item()

            if train_images > log_interval:
                total_train_images += train_images
                total_train_loss += train_loss
                if w_images == 0:
                    w_loss = train_loss
                    w_images = train_images
                log_message(None, "%3d %8d %8.3f %8.3f     %6.1f" % (
                e, total_train_images, train_loss / train_images, w_loss / w_images, (time.time() - start)))

                w_images = w_decay * w_images + train_images
                w_loss = w_decay * w_loss + train_loss
                train_images = 0
                train_loss = 0

        test_images = 0
        test_loss = 0

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader, 0):

                output = model(Variable(data))
                loss = criterion(output, Variable(data))

                test_images += len(data)
                test_loss += loss.data.item()


        log_message(log_file, "%3d %8d %8.3f %8.3f %8.3f   %6.1f" % (
        e, (e + 1) * total_train_images, total_train_loss / total_train_images, w_loss / w_images,
        test_loss / test_images, (time.time() - start)))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cifar-10 recognition starter code with LeNet (1998) CNN')
    parser.add_argument('-log', default='./log/trans-temp2.log', help='name of log file')
    parser.add_argument('-noisy', default=2, type=int, help='level of reporting')
    parser.add_argument('-path', default='./data/CIFAR', help='path of CIFAR data')
    # parser.add_argument('-path', required=True, help='path of CIFAR data')
    parser.add_argument('-save', default='./model/CIFAR/temp2.pth', help='saved model file')
    parser.add_argument('-batch', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=nEpochs, type=int, help='number of epochs')
    global log_file
    args = parser.parse_args()
    data_path = args.path
    batch_size = args.batch
    log_file = open(args.log, "a")
    nEpochs = args.epochs
    print("args =", args)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)  # ensure output directory exists
    log_file = open(args.log, "a")
    log_message(log_file, "\nstarting run: %s %s" % (datetime.date.today(), datetime.datetime.now()))

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=True, download=False,
                         transform=transforms.Compose(
                             [transforms.ToTensor(),
                              transforms.Normalize((mean, mean, mean), (std, std, std))])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=False,
                         transform=transforms.Compose(
                             [transforms.ToTensor(),
                              transforms.Normalize((mean, mean, mean), (std, std, std))])),
        batch_size=test_batch_size, shuffle=True)

    C0 = 3  # don't change
    C1 = 12
    C2 = 24
    C3 = 48
    kernel_size = 4

    padding = 1
    specs = [C0, C1, C2, C3, kernel_size, padding]
    model = Autoencoder(specs)

    model = train(model, train_loader, test_loader)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model, args.save)
