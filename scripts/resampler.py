import pandas as pd
import numpy as np
import os
import shutil

from pathlib import Path


data_dir = '/nfs/dataset/'
metadata_dir = '/nfs/metadata/'
new_data_dir = '/nfs/resampled_dataset/'

folders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
folders = np.array(folders)
print('Folders found: ', len(folders))

metadata = pd.read_csv(metadata_dir + 'compound_to_cell.csv')
metadata['Image_FileName_DAPI'] = metadata['Image_FileName_DAPI'].apply(lambda x: x.split('.tif')[0])
compound_folder = metadata[['Image_FileName_DAPI', 'Image_Metadata_Compound']].sort_values('Image_FileName_DAPI')
relevant_folders = compound_folder.Image_FileName_DAPI.isin(folders)
compound_folder = compound_folder[relevant_folders]
unique_compounds = compound_folder['Image_Metadata_Compound'].unique()
print(unique_compounds)
print('Unique Compounds found: ', len(unique_compounds))
print('Creating directories')
for compound in unique_compounds:
    compound = compound.replace('/', '-')
    folder = new_data_dir + compound
    print('Creating directory: ', folder)
    Path(folder).mkdir(parents=False, exist_ok=True)    
print('Directories created')

print('Moving files')
for compound in unique_compounds:
    print('Moving compound ', compound)
    folder_comp = compound.replace('/', '-')
    sub_folders = compound_folder.Image_FileName_DAPI[compound_folder['Image_Metadata_Compound'] == compound]
    dest_path = Path(new_data_dir + folder_comp + '/')
    i = 1
    for folder in sub_folders:
        src_dir = Path(data_dir + '/' + folder + '/')
        files = os.listdir(str(src_dir))
        for f in files:
            file_dir = Path(str(src_dir) + '/' + f)
            new_file = Path(str(dest_path) + '/' + str(i) + '.tif')
            shutil.copy(str(file_dir.absolute()), str(new_file.absolute()))
            i += 1
print('All files moved')

rng = np.random.default_rng(22)
rs_folders = os.listdir(new_data_dir)
for rsf in rs_folders:
    print('Resampling ', rsf)
    curr_dir = new_data_dir + rsf + '/'
    images = np.array(os.listdir(curr_dir))
    if rsf == 'DMSO':
        downsample = rng.choice(images, size=32600, replace=False)
        ds_mask = ~(np.isin(images, downsample))
        images = images[ds_mask]
        for f in images:
            Path(curr_dir + f).unlink()
    else:
        resample_size = 32600 - len(images)
        print('Resampling by ', resample_size, ' images')
        oversample = rng.choice(images, size=resample_size, replace=True)
        idx = len(images) + 1
        for f in oversample:
            src_path = Path(curr_dir + f)
            dest_path = Path(curr_dir + str(idx) + '.tif')
            shutil.copy(str(src_path.absolute()), str(dest_path.absolute()))
            idx += 1