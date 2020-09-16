import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
import torch
import umap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from torch import cuda
from torch.utils import data
from torchvision import datasets, transforms
from vae import VAE
from vae import loss_function as vae_loss
from vsc import VSC
from vsc import loss_function as vsc_loss

FORMAT = '%(asctime)s - %(levelname)-8s: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger('trainer')
logger.info('Starting evaluation')

base_dir = '/nfs/'

data_dir = base_dir + 'dataset/'
metadata_dir = base_dir + 'metadata/'
model_path = base_dir + 'runs/vscarch1/c250a0.05z256b1lr0.0001bs128/model/vsc50.pth'
results_path = base_dir + 'runs/demeaned_vsc_arch1/'

logger.info('Loading model:' + model_path)

Path(results_path).mkdir(parents=False, exist_ok=True)
logger.info(data_dir)
z = 256
batch_size = 512
alpha = 0.05
beta = 1
c = 250
whole_dataset = True
demean_data = True

torch.manual_seed(22)
np.random.seed(22)
device = torch.device("cuda" if cuda.is_available() else "cpu")
print(device)

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


if whole_dataset:
    image_dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]))
    cell_loader = data.DataLoader(
          image_dataset,
          batch_size=512,
          num_workers = 6,
          shuffle=True,
          pin_memory=True
        )
    label_mapping = image_dataset.class_to_idx
else:
    _, _, cell_loader, label_mapping = split_data(data_dir, batch_size=512, num_workers=6)

model = VSC(z, c)
loss_function = vsc_loss
model.load_state_dict(torch.load(model_path))
model.to(device)
print(model)

logger.info('Mapping metadata')
metadata = pd.read_csv(metadata_dir + 'compound_to_cell.csv')
metadata['Image_FileName_DAPI'] = metadata['Image_FileName_DAPI'].apply(lambda x: x.split('.tif')[0])
compound_folder = metadata[['Image_FileName_DAPI', 'Image_Metadata_Compound', 'Image_Metadata_Plate_DAPI']].sort_values('Image_FileName_DAPI')
cf_id = compound_folder['Image_FileName_DAPI'].apply(lambda x: int(label_mapping.get(x, -1)))
compound_folder['id'] = cf_id
compound_folder = compound_folder[compound_folder.id != -1]
comp_list = compound_folder['Image_Metadata_Compound'].unique()
indexed_compf = compound_folder.set_index('id')


model.eval()
total_batches = len(cell_loader)
batch_log = total_batches // 10
test_z = None
test_labels = []
logger.info('Getting latent space')
with torch.no_grad():
    for batch_id, (data, labels) in enumerate(cell_loader):
        data = data.to(device)
        encoded = model.encode(data)
        latent_data = model.reparameterize(*encoded)
        
        if test_z is not None:
            test_z = torch.cat((test_z, latent_data.cpu()), dim=0)
            test_labels = torch.cat((test_labels, labels))
        else:
            test_z = latent_data.cpu()
            test_labels = labels
            
        if batch_id % batch_log == 0:
            logger.info('Processing batch ' + str(batch_id) + '/' + str(total_batches))

test_z = test_z.numpy()
test_labels.numpy()
cell_df = indexed_compf.reindex(test_labels)
if demean_data:
    logger.info('Calculating DMSO mean per plate')
    dmso_slice = cell_df[cell_df.Image_Metadata_Compound == 'DMSO']
    dmso_plates = dmso_slice.Image_Metadata_Plate_DAPI.unique()
    dmso_mean = {}
    for plate in dmso_plates:
        plate_mask = (cell_df.Image_Metadata_Plate_DAPI == plate) & (cell_df.Image_Metadata_Compound == 'DMSO')
        plate_cells = test_z[plate_mask]
        plate_mean = np.mean(plate_cells, axis=0)
        dmso_mean[plate] = plate_mean

    logger.info('Control group mean calculated')

    logger.info('Demeaning Data')
    for plate in dmso_plates:
        plate_mask = cell_df.Image_Metadata_Plate_DAPI == plate
        test_z[plate_mask] -= dmso_mean[plate]
    logger.info('Data demeaned')

test_compounds = cell_df.Image_Metadata_Compound
del cell_df
logger.info('Getting Mahalanobis Distance')
def compound_mean(compound):
    is_compound = test_compounds == compound
    comp_matrix = test_z[is_compound]
    #covar = np.cov(comp_matrix.T)
    mean = np.mean(comp_matrix, axis=0)
    mean = mean.reshape(z,1)
    return mean

def mahalanobis_dist(x, y, covar): 
    delta = x - y
    mahalanobis = delta.T @ np.linalg.inv(covar) @ delta
    mahalanobis = np.sqrt(mahalanobis)
    return np.diag(mahalanobis)[0]

mc_cache = {}
covar = np.cov(test_z.T)
mb_matrix = np.empty(0)
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
    
    if mb_matrix.any():
        mb_matrix = np.vstack((mb_matrix, dist_array))
    else:
        mb_matrix = dist_array
        
plt.figure(figsize=(17,15))
fig = sns.heatmap(mb_matrix, cmap=plt.cm.viridis, annot=False)
plt.title('Mahalanobis Distance')
fig.set_xticklabels(comp_list, size=8)
fig.set_yticklabels(comp_list, size=8)
fig.figure.savefig(results_path + 'mahalanobis.png')
plt.close()

linkage = sch.linkage(mb_matrix, method='centroid', optimal_ordering=True)
dendro = sch.dendrogram(linkage, orientation='left', no_plot=True)

# Plot distance matrix.
labels = comp_list[dendro['leaves']]
omb = mb_matrix
index = dendro['leaves']
omb = omb[index,:]
omb = omb[:,index]
labels = comp_list[index]

plt.figure(figsize=(17,15))
fig = sns.heatmap(omb, xticklabels=labels, yticklabels=labels, cmap=plt.cm.viridis, annot=False)
plt.title('Mahalanobis Distance - SCH')
fig.set_xticklabels(labels, size=8)
fig.set_yticklabels(labels, size=8)
fig.figure.savefig(results_path + 'mahalanobis_sch.png')
plt.close()

logger.info("Mahalanobis Distance finished")
logger.info("Calculating Symmetric KL Distance")

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
i = 0
for x in comp_list:
    distances = []
    is_compound = test_compounds == x
    comp_x = test_z[is_compound]
    for y in comp_list:
        is_compound = test_compounds == y
        comp_y = test_z[is_compound]
        distances.append(symmetric_kl(comp_x, comp_y))
        
    dist_array = np.array(distances)
    if kldist_matrix.any():
        kldist_matrix = np.vstack((kldist_matrix, dist_array))
    else:
        kldist_matrix = dist_array
kldist_matrix.shape

plt.figure(figsize=(17,15))
fig = sns.heatmap(kldist_matrix, cmap=plt.cm.viridis, annot=False)
fig.set_xticklabels(comp_list, size=8)
fig.set_yticklabels(comp_list, size=8)
plt.title('Symmetric KLD')
fig.figure.savefig(results_path + 'sym_kld.png')
plt.close()

linkage = sch.linkage(kldist_matrix, method='centroid', optimal_ordering=True)
dendro = sch.dendrogram(linkage, orientation='left', no_plot=True)

# Plot distance matrix.
labels = comp_list[dendro['leaves']]
okl = kldist_matrix
index = dendro['leaves']
okl = okl[index,:]
okl = okl[:,index]
labels = comp_list[index]

plt.figure(figsize=(17,15))
fig = sns.heatmap(okl, xticklabels=labels, yticklabels=labels, cmap=plt.cm.viridis, annot=False)
fig.set_xticklabels(labels, size=8)
fig.set_yticklabels(labels, size=8)
fig.figure.savefig(results_path + 'sym_kld_sch.png')
plt.title('Symmetric KLD - SCH')
plt.close()

logger.info('Evaluating classification performance')

clf_train_idx = int(len(test_z) * 0.8)
clf_train_split = test_z[:clf_train_idx]
clf_train_labels = test_compounds[:clf_train_idx]

clf_val_split = test_z[clf_train_idx:]
clf_val_labels = test_compounds[clf_train_idx:]

logger.info('Running Stratified Dummy')
classifier = DummyClassifier(strategy='stratified', random_state=22)
classifier.fit(clf_train_split, clf_train_labels)
pred = classifier.predict(clf_val_split)
report = classification_report(clf_val_labels, pred, digits=4)
with open(results_path + 'dummy_report.txt', "w") as text_file:
    print(report, file=text_file)

logger.info('Running Naive Bayes Classifier')
classifier = GaussianNB()
classifier.fit(clf_train_split, clf_train_labels)
pred = classifier.predict(clf_val_split)
report = classification_report(clf_val_labels, pred, digits=4)
with open(results_path + 'naivebayes_report.txt', "w") as text_file:
    print(report, file=text_file)

logger.info('Running Quadratic Discriminant Analysis Classifier')
classifier = QuadraticDiscriminantAnalysis()
classifier.fit(clf_train_split, clf_train_labels)
pred = classifier.predict(clf_val_split)
report = classification_report(clf_val_labels, pred, digits=4)
with open(results_path + 'qda_report.txt', "w") as text_file:
    print(report, file=text_file)

logger.info('Running MLP Classifier')
classifier = MLPClassifier(hidden_layer_sizes=(280, 200, 150), random_state=22,
                           max_iter=300, early_stopping=True, verbose=True)
classifier.fit(clf_train_split, clf_train_labels)
pred = classifier.predict(clf_val_split)
report = classification_report(clf_val_labels, pred, digits=4)
with open(results_path + 'mlp_report.txt', "w") as text_file:
    print(report, file=text_file)

del classifier

logger.info('Classification done')
logger.info('Projecting with UMAP')
if whole_dataset:
    subset_idx = int(len(test_z) * 0.7)
    test_z = test_z[:subset_idx]
    test_compounds = test_compounds[:subset_idx]

reducer = umap.UMAP(random_state=22, verbose=True, low_memory=True)
z_2d = reducer.fit_transform(test_z)

plt.figure(figsize=(10,10))
fig = sns.scatterplot(z_2d[:,0], z_2d[:,1], alpha=0.5,size=1, hue=test_compounds, palette=sns.hls_palette(len(comp_list)),legend=False)
plt.title('All Compounds')
fig.figure.savefig(results_path + 'scatter_all.png')
plt.close()

plt.figure(figsize=(10,10))
plt.title('All compounds')
fig = sns.kdeplot(z_2d[:, 0], z_2d[:, 1], legend=True, shade=True, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
fig.figure.savefig(results_path + 'kde_all.png')
plt.close()

logger.info('Creating UMAP plots')
for compound in comp_list:
    #selected_comps = compound_folder[compound_folder['Image_Metadata_Compound'] == compound]
    #is_compound = np.isin(test_labels, selected_comps.id.values)
    is_compound = test_compounds == compound
    comp_slice = z_2d[is_compound]
    x = comp_slice[:, 0]
    y = comp_slice[:, 1]

    compound = compound.replace('/', '-')

    plt.figure(figsize=(10, 10))
    plt.title(compound + ' UMAP')
    fig = sns.scatterplot(x, y, alpha=0.7, size=1, palette=sns.cubehelix_palette())
    fig.figure.savefig(results_path + 'scatter_' + compound + '.png')
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.title(compound + ' Density Plot')
    fig = sns.kdeplot(x, y, legend=True, shade=True, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
    fig.figure.savefig(results_path + 'kde_' + compound + '.png')
    plt.close()

logger.info('Eval finished')