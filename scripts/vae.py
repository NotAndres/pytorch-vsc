import argparse
import json
import os
import time
import logging
from os import listdir
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import cuda
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

FORMAT = '%(asctime) - [%(level)-8s]: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('trainer')

# First some utilities so saving everything is easier later
# Also this makes it easier to run the model with different hyper parameters

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', help='Directory where results will be stored', default='')
parser.add_argument('--dataset_dir', help='Path to the dataset to use')
parser.add_argument('--beta', help='Beta value for the KLD', type=int, default=1)
parser.add_argument('--lr', help='Learning rate to be used in the optimizer', type=float, default=1e-4)
parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=50)
parser.add_argument('--batch_size', help='Size of mini batches', type=int, default=128)
parser.add_argument('--checkpoints', help='Store model every 10 epochs', type=bool, default=True)
parser.add_argument('--parallel', help='Set true for multiple gpus', type=bool, default=False)
parser.add_argument('--num_workers', help="Num of workers (threads) that will retrieve the data", type=int, default=0)
parser.add_argument('--resume', help="Resume training if there's a checkpoint available", type=bool, default=False)
args = parser.parse_args()

base_dir = args.base_dir
beta = args.beta
epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size
data_dir = args.dataset_dir
checkpoints = args.checkpoints
parallel = args.parallel
workers = args.num_workers
resume = args.resume
# Make this an arg later
latent_dim = 256
logger.info(args)

if base_dir and not os.path.exists(base_dir):
    raise FileNotFoundError("Base directory doesnt exist")

if data_dir and not os.path.exists(data_dir):
    raise FileNotFoundError("Invalid path to dataset")

folder_name = '/epochs' + str(epochs) + 'beta' + str(beta) + "lr" + str(learning_rate) + "/"
recon_folder = base_dir + folder_name + 'reconstruction/'
samples_folder = base_dir + folder_name + 'samples/'
model_folder = base_dir + folder_name + 'model/'
tensorboard_folder = base_dir + folder_name + 'tb/'

Path(recon_folder).mkdir(parents=True, exist_ok=True)
Path(samples_folder).mkdir(parents=False, exist_ok=True)
Path(model_folder).mkdir(parents=False, exist_ok=True)
Path(tensorboard_folder).mkdir(parents=False, exist_ok=True)
# Setting manual seed for reproducibility
torch.manual_seed(22)

# Is cuda available?
if not cuda.is_available():
    raise RuntimeError('Cuda is not available')
device = torch.device("cuda")

# TensorBoard
tb_writer = SummaryWriter(tensorboard_folder)


# Model for the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv1 = self.getConvolutionLayer(3, 128)
        self.encoder_conv2 = self.getConvolutionLayer(128, 64)
        self.encoder_conv3 = self.getConvolutionLayer(64, 32)

        self.flatten = nn.Flatten()

        self.encoder_fc1 = nn.Linear(4608, self.latent_dim)
        self.encoder_fc2 = nn.Linear(4608, self.latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, 4608),
            nn.ReLU()
        )
        # Reshape to 32x12x12
        self.decoder_upsampler1 = nn.Upsample(scale_factor=(2, 2), mode='nearest')

        self.decoder_deconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=(2, 2), mode='nearest')
        )
        # 48x48x64
        self.decoder_deconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=(2, 2), mode='nearest')
        )

        self.decoder_conv1 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        # 96x96x128

    def getConvolutionLayer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def encode(self, x):
        x = self.encoder_conv1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_conv3(x)

        x = self.flatten(x)
        mu = self.encoder_fc1(x)
        sigma = self.encoder_fc2(x)

        return mu, sigma

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # Keeps shape, samples from normal dist with mean 0 and variance 1
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_fc1(z)
        z = self.decoder_upsampler1(z.view(-1, 32, 12, 12))
        z = self.decoder_deconv1(z)
        z = self.decoder_deconv2(z)
        recon = self.decoder_conv1(z)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Loss Function definition
def loss_function(recon_x, x, mu, logvar):
    mse = torch.mean(torch.sum((x - recon_x).pow(2), dim=(1, 2, 3)))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta

    loss = mse + kld
    return loss, mse, kld


# Image processing utilities
dataset_mean = (0.0302, 0.0660, 0.0518)
dataset_std = (0.0633, 0.0974, 0.0766)


def split_data(data_dir, n_split=0.2, batch_size=256, num_workers=0):
    pin_memory = cuda.is_available()
    # Create training and validation datasets
    image_dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ]))
    # Create training and validation dataloaders
    total = len(image_dataset)
    n_test = int(total * n_split)
    n_train = total - n_test
    train_set, test_set = data.random_split(image_dataset, (n_train, n_test))

    n_val = int(len(train_set) * n_split)
    n_train = len(train_set) - n_val
    train_set, val_set = data.random_split(train_set, (n_train, n_val))

    logger.info('Train split: ', len(train_set))
    logger.info('Val split: ', len(val_set))
    logger.info('Test split: ', len(test_set))

    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader


# Make tensor into a grid of images and then remove normalization
def grid_and_unnormalize(batch_tensor):
    grid = torchvision.utils.make_grid(batch_tensor)
    image = grid.numpy().transpose((1, 2, 0))
    mean = np.asarray(dataset_mean)
    std = np.asarray(dataset_std)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


# Training loop helpers
def get_time_in_hours(seconds):
    hours = seconds // 3600
    remaining_time = seconds % 3600
    minutes = remaining_time // 60
    seconds = remaining_time % 60

    return hours, minutes, seconds


def train(model, optimizer, epoch, train_loader):
    model.train()
    train_loss = 0
    train_mse = 0
    train_kld = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, mse, kld = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()

        current_batch_size = len(data)
        train_loss += loss.item() * current_batch_size
        train_mse += mse.item() * current_batch_size
        train_kld += kld.item()

        if batch_idx % (int(len(train_loader) / 4)) == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item()))

    datapoints = len(train_loader.dataset)
    avg_loss = train_loss / datapoints
    avg_mse = train_mse / datapoints
    avg_kld = train_kld / (beta * len(train_loader))

    logger.info('====> Epoch: {} Average loss: {:.8f}'.format(epoch, avg_loss))
    logger.info('*** Avg MSE: {:.4f}'.format(avg_mse))
    logger.info('*** Avg KLD: {:.8f}'.format(avg_kld * beta))
    return avg_loss, avg_mse, avg_kld


def test(model, epoch, test_loader):
    model.eval()
    test_loss = 0
    test_mse = 0
    test_kld = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, mse, kld = loss_function(recon_batch, data, mu, logvar)

            current_batch_size = data.size(0)
            test_loss += loss.item() * current_batch_size
            test_mse += mse.item() * current_batch_size
            test_kld += kld.item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch[:n]]).cpu()
                comparison = grid_and_unnormalize(comparison)
                tb_writer.add_image("Reconstruction/recon_" + str(epoch), comparison, epoch, dataformats='HWC')

                plt.imshow(comparison)
                plt.savefig(recon_folder + 'reconstruction_' + str(epoch) + '.png')
                plt.close()

    datapoints = len(test_loader.dataset)
    test_loss /= datapoints
    test_mse /= datapoints
    test_kld /= (beta * len(test_loader))
    logger.info('====> Test set loss: {:.8f}'.format(test_loss))
    logger.info('*** Avg MSE: {:.8f}'.format(test_mse))
    logger.info('*** Avg KLD: {:.8f}'.format(test_kld))
    return test_loss, test_mse, test_kld


# Training starts here
# Splitting data
logger.info('** Splitting dataset')
train_data, val_data, test_data = split_data(data_dir=data_dir, batch_size=batch_size, num_workers=workers)

# Model and optimizer
model = VAE(latent_dim)

if parallel and torch.cuda.device_count() > 1:
    logger.info("Setting model for multiple GPUs")
    logger.info("GPUs: " + str(torch.cuda.device_count()))
    model = nn.DataParallel(model)
else:
    parallel = False

starting_epoch = 1
if resume:
    logger.info('===> Loading Checkpoint <===')
    cp_name = max(listdir(model_folder))
    logger.info('Latest checkpoint found: ' + cp_name)
    model.load_state_dict(torch.load(model_folder + cp_name))
    starting_epoch = int(cp_name[3:5]) + 1
    logger.info('===> Checkpoint Loaded <===')

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# dicts to keep track of loss
train_trace = {
    'loss': [],
    'mse': [],
    'kld': []
}

val_trace = {
    'loss': [],
    'mse': [],
    'kld': []
}

# Training loop
since = time.time()

for epoch in range(starting_epoch, epochs + 1):
    train_loss, train_mse, train_kld = train(model, optimizer, epoch, train_data)
    train_trace['loss'].append(train_loss)
    train_trace['mse'].append(train_mse)
    train_trace['kld'].append(train_kld)

    val_loss, val_mse, val_kld = test(model, epoch, val_data)
    val_trace['loss'].append(val_loss)
    val_trace['mse'].append(val_mse)
    val_trace['kld'].append(val_kld)

    tb_writer.add_scalars('Loss/Total', {
        'train': train_loss,
        'test': val_loss
    }, epoch)

    tb_writer.add_scalars('Loss/MSE', {
        'train': train_mse,
        'test': val_mse
    }, epoch)

    tb_writer.add_scalars('Loss/KLD', {
        'train': train_kld,
        'test': val_kld
    }, epoch)

    with torch.no_grad():
        sample = torch.randn(4, 256).to(device)
        if parallel:
            sample = model.module.decode(sample).cpu()
        else:
            sample = model.decode(sample).cpu()
        sample = grid_and_unnormalize(sample)
        tb_writer.add_image("Reconstruction/recon_" + str(epoch), sample, epoch, dataformats='HWC')

        plt.imshow(sample)
        plt.savefig(samples_folder + 'sample_' + str(epoch) + '.png')
        plt.close()

    # Save model every 10 epochs
    if checkpoints and (epoch % 10 == 0):
        logger.info("===> Saving checkpoint <===")
        model_name = model_folder + 'vae' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_name)

    epoch_time = time.time() - since
    e_hours, e_minutes, e_seconds = get_time_in_hours(epoch_time)
    logger.info('Time elapsed {:.0f}h {:.0f}m {:.0f}s'.format(e_hours, e_minutes, e_seconds))

logger.info("**** Saving loss data ****")
trace = {'train': train_trace, 'validation': val_trace}
with open(base_dir + folder_name + 'loss.json', 'w') as file:
    json.dump(trace, file)

tb_writer.close()
logger.info("Training finished")
