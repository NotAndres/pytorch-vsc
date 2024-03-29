import argparse
import json
import logging
import os
import time
from os import listdir
from pathlib import Path

from vae import VAE
from vae import loss_function as vae_loss
from vsc import VSC
from vsc import loss_function as vsc_loss

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
import umap
from torch import cuda
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

FORMAT = '%(asctime)s - %(levelname)-8s: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger('trainer')

# First some utilities so saving everything is easier later
# Also this makes it easier to run the model with different hyper parameters

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', help='Directory where results will be stored', default='')
parser.add_argument('--dataset_dir', help='Path to the dataset to use')
parser.add_argument('--metadata_dir', help='Path to metadata files')
parser.add_argument('--model', help="Model to use for training", type=str, default='vae')

parser.add_argument('--latent_dim', help='Size of the latent space', type=int, default=256)
parser.add_argument('--beta', help='Beta value for the KLD', type=int, default=1)
parser.add_argument('--lr', help='Learning rate to be used in the optimizer', type=float, default=1e-4)
parser.add_argument('--epochs', help='Number of epochs to train for', type=int, default=50)
parser.add_argument('--batch_size', help='Size of mini batches', type=int, default=128)
parser.add_argument('--alpha', help='Alpha value for VSC', type=float, default=0.5)
parser.add_argument('--c', help='Starting C value', type=int, default=50)

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
# VSC params
c = 50
end_c = args.c
alpha = args.alpha

data_dir = args.dataset_dir
metadata_dir = args.metadata_dir

checkpoints = args.checkpoints
parallel = args.parallel
workers = args.num_workers
resume = args.resume
# Make this an arg later
latent_dim = args.latent_dim
model_type = args.model
logger.info(args)

if base_dir and not os.path.exists(base_dir):
    raise FileNotFoundError("Base directory doesnt exist")

if data_dir and not os.path.exists(data_dir):
    raise FileNotFoundError("Invalid path to dataset")

if metadata_dir and not os.path.exists(metadata_dir):
    raise FileNotFoundError("Invalid path to metadata")

folder_name = '/' + model_type + 'arch1' + '/'
is_vsc = model_type == 'vsc'
if is_vsc:
    folder_name += 'c' + str(end_c) + 'a' + str(alpha)
folder_name += 'z' + str(latent_dim) + 'b' + str(beta) + 'lr' + str(learning_rate) + 'bs' + str(batch_size) + '/'

recon_folder = base_dir + folder_name + 'reconstruction/'
samples_folder = base_dir + folder_name + 'samples/'
model_folder = base_dir + folder_name + 'model/'
tensorboard_folder = base_dir + '/tensorboard/' + folder_name

Path(recon_folder).mkdir(parents=True, exist_ok=True)
Path(samples_folder).mkdir(parents=False, exist_ok=True)
Path(model_folder).mkdir(parents=False, exist_ok=True)
Path(tensorboard_folder).mkdir(parents=True, exist_ok=True)
# Setting manual seed for reproducibility
batch_size = 512
torch.manual_seed(22)

# Is cuda available?
if not cuda.is_available():
    raise RuntimeError('Cuda is not available')
device = torch.device("cuda")

# TensorBoard
tb_writer = SummaryWriter(tensorboard_folder)

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

    logger.info('Train split: ' + str(len(train_set)))
    logger.info('Val split: ' + str(len(val_set)))
    logger.info('Test split: ' + str(len(test_set)))

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
    return train_loader, val_loader, test_loader, image_dataset.class_to_idx


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
    train_slab = 0
    train_spike = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        current_batch_size = data.size(0)
        fwd_args = model(data)
        if is_vsc:
            recon_batch, mu, logvar, gamma = fwd_args
            loss, mse, kld, slab, spike = vsc_loss(recon_batch, data, mu, logvar, gamma, alpha=alpha)
            train_slab += slab.item()
            train_spike += spike.item()
        else:
            recon_batch, mu, logvar = fwd_args
            loss, mse, kld = vae_loss(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()

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
    if is_vsc:
        train_slab /= (beta * len(train_loader))
        train_spike /= (beta * len(train_loader))
        logger.info('* Avg Slab: {:.8f}'.format(train_slab))
        logger.info('* Avg Spike: {:.8f}'.format(train_spike))

    return avg_loss, avg_mse, avg_kld, train_slab, train_spike


def test(model, epoch, test_loader):
    model.eval()
    test_loss = 0
    test_mse = 0
    test_kld = 0
    test_slab = 0
    test_spike = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            current_batch_size = data.size(0)
            fwd_args = model(data)
            if is_vsc:
                recon_batch, mu, logvar, gamma = fwd_args
                loss, mse, kld, slab, spike = vsc_loss(recon_batch, data, mu, logvar, gamma, alpha=alpha)
                test_slab += slab.item()
                test_spike += spike.item()
            else:
                recon_batch, mu, logvar = fwd_args
                loss, mse, kld = vae_loss(recon_batch, data, mu, logvar)

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
    if is_vsc:
        test_slab /= (beta * len(test_loader))
        test_spike /= (beta * len(test_loader))
        logger.info('* Avg Slab: {:.8f}'.format(test_slab))
        logger.info('* Avg Spike: {:.8f}'.format(test_spike))
    return test_loss, test_mse, test_kld, test_slab, test_spike


# Training starts here
# Splitting data
logger.info('** Splitting dataset')
train_data, val_data, test_data, label_mapping = split_data(data_dir=data_dir, batch_size=batch_size, num_workers=workers)

# Model and optimizer
model = None
if is_vsc:
    model = VSC(latent_dim, c)
else:
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
    cp_name = listdir(model_folder)
    model_number = max([int(item.split('.pth')[0][3:]) for item in cp_name])
    cp_name = model_type + str(model_number) + '.pth'

    logger.info('Latest checkpoint found: ' + cp_name)
    model.load_state_dict(torch.load(model_folder + cp_name))
    starting_epoch = model_number + 1
    logger.info('===> Checkpoint Loaded <===')
    denormal = torch.set_flush_denormal(True)
    logger.info('Flushing denormal:' + str(denormal))

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# dicts to keep track of loss
keys = ['Total', 'MSE', 'KLD']
if is_vsc:
    keys.append('Slab')
    keys.append('Spike')
    model.update_c(starting_epoch - 1 + c)
train_trace = {}
val_trace = {}
for key in keys:
    train_trace[key] = []
    val_trace[key] = []

# Training loop
since = time.time()

c_step = (end_c - c) / epochs
c = c + c_step * (starting_epoch - 1)

for epoch in range(starting_epoch, epochs + 1):

    if is_vsc:
        c += c_step
        model.update_c(c)

    train_loss = train(model, optimizer, epoch, train_data)
    val_loss = test(model, epoch, val_data)

    for idx, term in enumerate(keys):
        train_trace[term].append(train_loss[idx])
        val_trace[term].append(val_loss[idx])

        tb_writer.add_scalars('Loss/' + term, {
            'train': train_loss[idx],
            'test': val_loss[idx],
        }, epoch)

    with torch.no_grad():
        sample = torch.randn(4, latent_dim).to(device)
        if parallel:
            sample = model.module.decode(sample).cpu()
        else:
            sample = model.decode(sample).cpu()
        sample = grid_and_unnormalize(sample)
        tb_writer.add_image("Sample/sample_" + str(epoch), sample, epoch, dataformats='HWC')

        plt.imshow(sample)
        plt.savefig(samples_folder + 'sample_' + str(epoch) + '.png')
        plt.close()

    # Save model every 10 epochs
    if checkpoints and (is_vsc or (epoch % 10 == 0)):
        logger.info("===> Saving checkpoint <===")
        model_name = model_folder + model_type + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_name)

    epoch_time = time.time() - since
    e_hours, e_minutes, e_seconds = get_time_in_hours(epoch_time)
    logger.info('Time elapsed {:.0f}h {:.0f}m {:.0f}s'.format(e_hours, e_minutes, e_seconds))

if starting_epoch < epochs:
    logger.info("**** Saving loss data ****")
    trace = {'train': train_trace, 'validation': val_trace}
    with open(base_dir + folder_name + 'loss.json', 'w') as file:
        json.dump(trace, file)
    logger.info("Training finished")

logger.info('===> Starting Evaluation <===')

# Loading the metadata file and slicing to those folders/labels that are available
metadata = pd.read_csv(metadata_dir + 'compound_to_cell.csv')
metadata['Image_FileName_DAPI'] = metadata['Image_FileName_DAPI'].apply(lambda x: x.split('.tif')[0])
compound_folder = metadata[['Image_FileName_DAPI', 'Image_Metadata_Compound']].sort_values('Image_FileName_DAPI')
cf_id = compound_folder['Image_FileName_DAPI'].apply(lambda x: int(label_mapping.get(x, -1)))
compound_folder['id'] = cf_id
compound_folder = compound_folder[compound_folder.id != -1]
comp_list = compound_folder['Image_Metadata_Compound'].unique()
#comp_list = np.array(label_mapping.keys())

model.eval()
if parallel:
    model = model.module

# Getting Z for the test set
test_z = None
test_labels = None
test_loss = 0
test_mse = 0
test_kld = 0
test_slab = 0
test_spike = 0
with torch.no_grad():
    total_batches = len(test_data)
    for batch_idx, (data, labels) in enumerate(test_data):
        data = data.to(device)
        encode_args = model.encode(data)
        batch_z = model.reparameterize(*encode_args)
        recon_x = model.decode(batch_z)
        current_batch_size = data.size(0)
        if is_vsc:
            loss, mse, kld, slab, spike = vsc_loss(recon_x, data, *encode_args, alpha=alpha)
            test_slab += slab
            test_spike += spike
        else:
            loss, mse, kld = vae_loss(recon_x, data, *encode_args, beta=beta)

        test_loss += loss.item() * current_batch_size
        test_mse += mse.item() * current_batch_size
        test_kld += kld.item()

        if batch_idx % (total_batches // 10):
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_x[:n]]).cpu()
            comparison = grid_and_unnormalize(comparison)
            tb_writer.add_image("Eval/Reconstruction/recon" + str(batch_idx//(total_batches // 10)),
                                comparison,
                                dataformats='HWC')
        batch_z = batch_z.cpu()
        if test_z is not None:
            test_z = torch.cat((test_z, batch_z), dim=0)
            test_labels = torch.cat((test_labels, labels))
        else:
            test_z = batch_z
            test_labels = labels

datapoints = len(test_data.dataset)
test_loss /= datapoints
test_mse /= datapoints
test_kld /= (beta * len(test_data))
logger.info('===> Evaluation loss: {:.8f}'.format(test_loss))
logger.info('*** Avg MSE: {:.8f}'.format(test_mse))
logger.info('*** Avg KLD: {:.8f}'.format(test_kld))
if is_vsc:
    test_slab /= (beta * len(test_data))
    test_spike /= (beta * len(test_data))
    logger.info('* Avg Slab: {:.8f}'.format(test_slab))
    logger.info('* Avg Spike: {:.8f}'.format(test_spike))

#label_mapping = {v: k for k, v in label_mapping.items()}
#comp_list = np.array(list(label_mapping.values()))

test_compounds = []
for label in test_labels:
    compound = compound_folder[compound_folder.id == label.item()]['Image_Metadata_Compound'].values[0]
    test_compounds.append(compound)
    #test_compounds.append(label_mapping[label.item()])
test_compounds = np.array(test_compounds)
categorical_labels = pd.Series(test_compounds, dtype="category")

folder_name = base_dir + folder_name
sns.set(font_scale=0.5)

logger.info('Calculating Mahalanobis distance')


# Mahalanobis distance
def compound_mean(compound):
    selected_comps = compound_folder[compound_folder['Image_Metadata_Compound'] == compound]
    is_compound = np.isin(test_labels, selected_comps.id.values)
    #is_compound = test_compounds == compound
    comp_matrix = test_z[is_compound].numpy()
    mean = np.mean(comp_matrix, axis=0)
    mean = mean.reshape(latent_dim, 1)
    return mean


def mahalanobis_dist(x, y, covar):
    delta = x - y
    mahalanobis = delta.T @ np.linalg.inv(covar) @ delta
    mahalanobis = np.sqrt(mahalanobis)
    return np.diag(mahalanobis)[0]


mc_cache = {}
covar = np.cov(test_z.numpy().T)
dist_matrix = np.empty(0)
for base in comp_list:
    distances = []
    base_mean = None

    if base in mc_cache.keys():
        base_mean = mc_cache[base]
    else:
        base_mean = compound_mean(base)
        mc_cache[base] = base_mean

    for target in comp_list:
        target_mean = None
        target_covar = None

        if target in mc_cache.keys():
            target_mean = mc_cache[target]
        else:
            target_mean = compound_mean(target)
            mc_cache[target] = target_mean

        mahalanobis = mahalanobis_dist(base_mean, target_mean, covar)
        distances.append(mahalanobis)

    dist_array = np.array(distances)

    if dist_matrix.any():
        dist_matrix = np.vstack((dist_matrix, dist_array))
    else:
        dist_matrix = dist_array

plt.figure(figsize=(12, 10))
hm = sns.heatmap(dist_matrix, xticklabels=comp_list, yticklabels=comp_list, cmap=plt.cm.viridis)
plt.title('Mahalanobis Distance')
md_figname = folder_name + 'mahalanobis.png'
hm.figure.savefig(md_figname)
tb_writer.add_figure('Distances/Mahalanobis', hm.figure, global_step=epochs)

del mc_cache
del covar
del dist_matrix

logger.info('Calculating Symmetric KLD')


# KLD
def std_matrix(x):
    sigma = np.eye(len(x[0]))
    std = np.std(x, axis=0)
    np.fill_diagonal(sigma, std)
    return sigma


def kl_distance(x, y):
    sigma_x = std_matrix(x)
    sigma_y = std_matrix(y)
    d = len(x[0])
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)

    kl = np.trace(np.linalg.inv(sigma_y) @ sigma_x)
    kl += ((mu_y - mu_x).T @ np.linalg.inv(sigma_y) @ (mu_y - mu_x)) - d
    kl += np.log(np.linalg.det(sigma_y) / np.linalg.det(sigma_x))
    return kl


def symmetric_kl(x, y):
    kl_xy = kl_distance(x, y)
    kl_yx = kl_distance(y, x)

    return 0.5 * (kl_xy + kl_yx)


kldist_matrix = np.empty(0)
for x in comp_list:
    distances = []
    selected_comps = compound_folder[compound_folder['Image_Metadata_Compound'] == x]
    is_compound = np.isin(test_labels, selected_comps.id.values)
    #is_compound = test_compounds == x
    comp_x = test_z[is_compound].numpy()
    for y in comp_list:
        selected_comps = compound_folder[compound_folder['Image_Metadata_Compound'] == y]
        is_compound = np.isin(test_labels, selected_comps.id.values)
        #is_compound = test_compounds == y
        comp_y = test_z[is_compound].numpy()
        distances.append(symmetric_kl(comp_x, comp_y))

    dist_array = np.array(distances)
    if kldist_matrix.any():
        kldist_matrix = np.vstack((kldist_matrix, dist_array))
    else:
        kldist_matrix = dist_array

plt.figure(figsize=(12, 10))
hm = sns.heatmap(kldist_matrix, xticklabels=comp_list, yticklabels=comp_list, cmap=plt.cm.viridis)
plt.title('Compound Symmetric KLD')
kld_figname = folder_name + 'kld.png'
hm.figure.savefig(kld_figname)
tb_writer.add_figure('Distances/Symmetric_KL', hm.figure, global_step=epochs)

# Interpolation test
idx = np.argmax(kldist_matrix)
s, _ = kldist_matrix.shape
comp_a = idx // s
comp_b = idx % s
comp_a = comp_list[comp_a]
comp_b = comp_list[comp_b]

selected_comps = compound_folder[compound_folder['Image_Metadata_Compound'] == comp_a]
is_compound = np.isin(test_labels, selected_comps.id.values)
#is_compound = test_compounds == comp_a
comp_arr_a = test_z[is_compound].numpy()

selected_comps = compound_folder[compound_folder['Image_Metadata_Compound'] == comp_b]
is_compound = np.isin(test_labels, selected_comps.id.values)
#is_compound = test_compounds == comp_b
comp_arr_b = test_z[is_compound].numpy()

np.random.seed(22)
base_idx = np.random.randint(len(comp_arr_a))
target_idx = np.random.randint(len(comp_arr_b))

base = torch.from_numpy(comp_arr_a[base_idx:base_idx+1]).to(device)
target = torch.from_numpy(comp_arr_b[target_idx:target_idx+1]).to(device)

diff = target - base
recon_25 = model.decode(base + diff*0.25)
recon_50 = model.decode(base + diff*0.50)
recon_75 = model.decode(base + diff*0.75)
base = model.decode(base)
target = model.decode(target)
interpolation = torch.cat((base, recon_25, recon_50, recon_75, target), dim=0).cpu().detach()
interpolation = grid_and_unnormalize(interpolation)
tb_writer.add_image("Eval/Interpolation/ip", interpolation, dataformats='HWC', global_step=epochs)

del kldist_matrix


# 2D Projection
logger.info('Projecting with UMAP')
reducer = umap.UMAP(low_memory=True, random_state=22)
z_umap = reducer.fit_transform(test_z)
logger.info('UMAP finished')
plt.figure(figsize=(10,10))
scatter = sns.scatterplot(z_umap[:, 0], z_umap[:, 1], alpha=0.9, size=1, hue=categorical_labels.cat.codes, palette=sns.hls_palette(len(comp_list)), legend=False)
plt.title('All compounds')
tb_writer.add_figure('Projection/UMAP', scatter.figure, global_step=epochs)

plt.figure(figsize=(10,10))
plt.title('All compounds')
ax = sns.kdeplot(z_umap[:, 0], z_umap[:, 1], legend=True, shade=True, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
tb_writer.add_figure('Projection/DensityPlot', ax.figure, global_step=epochs)


# Plotting scatter and density plots
logger.info('Creating UMAP plots')
for compound in comp_list:
    selected_comps = compound_folder[compound_folder['Image_Metadata_Compound'] == compound]
    is_compound = np.isin(test_labels, selected_comps.id.values)
    #is_compound = test_compounds == compound
    comp_slice = z_umap[is_compound]
    x = comp_slice[:, 0]
    y = comp_slice[:, 1]

    plt.figure(figsize=(10, 10))
    plt.title(compound + ' UMAP')
    ax = sns.scatterplot(x, y, alpha=0.7, size=1, palette=sns.cubehelix_palette())
    tb_writer.add_figure('Scatterplot/' + compound, ax.figure, global_step=epochs)
    plt.figure(figsize=(10, 10))
    plt.title(compound + ' Density Plot')
    ax = sns.kdeplot(x, y, legend=True, shade=True, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
    tb_writer.add_figure('DensityPlot/' + compound, ax.figure, global_step=epochs)

logger.info('Evaluation finished')
tb_writer.close()

