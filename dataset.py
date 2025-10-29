import pandas as pd
from einops import rearrange
import torch
import ants
import os
import numpy as np
from monai.transforms import (Resize, Compose, Flip, NormalizeIntensity)


def _train_test_split(file_list, fold, type='train'):
    test_fold = fold
    # print(file_list)
    file_list = pd.read_csv(file_list)

    if type == 'train':
        file_list = file_list.loc[file_list['fold'] != test_fold]
    elif type == 'val':
        print('validation phase with fold: {}'.format(test_fold))
        file_list = file_list.loc[file_list['fold'] == test_fold]

    return file_list


def _get_slice_and_convert_to_tensor(img, z):#tranpose
    img = rearrange(img, 'h w d -> d h w')
    slices = img[z:z + 1, :, :]
    return torch.from_numpy(slices).type(torch.float32)

def _get_all_slices(img):
    img = rearrange(img, 'h w d -> d h w')
    # slices = [img[i:i+1] for i in range(img.shape[0])] 
    # print(len(slices))  
    # print(slices[0].shape) 
    slices = [torch.tensor(img[i:i + 1], dtype=torch.float32) for i in range(img.shape[0])]
    return slices

def _get_all_slices_nol(img):
    img = rearrange(img, 'h w d -> d h w')
    # slices = [img[i:i+1] for i in range(img.shape[0])]
    # for i in range(len(slices)):
    #     slices[i] = np.where(slices[i] == 0, 1.0, slices[i])
    slices = [torch.tensor(img[i:i + 1], dtype=torch.float32) for i in range(img.shape[0])]  
    for i in range(len(slices)):
    
        slices[i] = torch.where(slices[i] == 0, torch.tensor(0.1, dtype=torch.float32), slices[i])
    return slices

def _get_slices_and_convert_to_tensor(img, z_start, z_end):  
    img = rearrange(img, 'h w d -> d h w')
    slices = img[z_start:z_end, :, :] 
    return torch.from_numpy(slices).type(torch.float32)  


def _ants_img_info(img_path):
    """Function to get medical image information"""
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()
    #return image.GetOrigin(), image.GetSpacing(), image.GetDirection(), sitk.GetArrayFromImage(image)


file_list_path = '/home_data/home/v-wangtong/Dataset_for_VCDM/file_list.csv'
file_list = _train_test_split(file_list_path, 1, 'train')
# print(file_list)
# print(file_list.iloc[0][0]) # BAI_FENG_GANG
# print(file_list.iloc[166][0]) # ZHOU_TAO

class MD301Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, phase, mode, **kwargs):
        self._name = 'MD301Dataset'

        self.phase = 'train'
        self.root = dataroot
        self.mode = mode
        # self.file_list = os.path.join(dataroot, 'file_list.csv')
        self.file_list = file_list
        self._resolution = 448 

        # self.file_list = _train_test_split(file_list, 1, phase)
        #self.transform = NormalizeIntensity()
        #self.transform_upsample = Compose([Resize([896, 896]), NormalizeIntensity()])
        self.transform_upsample = Compose(Resize([448, 448]))
        # print(self.file_list)
        # print(len(self.file_list))


    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """


        # self.file_list = _train_test_split(self.file_list, 1, self.phase)  
        file_name = self.file_list.iloc[idx][0]
        # slice_idx = idx % 60 
        # offset = 10

        # Read volume images
        img_path_3T = os.path.join(self.root, '3T', file_name + '.nii.gz')
        _, _, _, img_3T = _ants_img_info(img_path_3T)
        # img_slices_3T = self.transform_upsample(_get_all_slices(img_3T))
        img_slices_3T = _get_all_slices(img_3T)

        img_path_7T = os.path.join(self.root, '7T_downsample_normal', file_name + '.nii.gz')
        _, _, _, img_7T = _ants_img_info(img_path_7T)
        img_slices_7T = _get_all_slices(img_7T)#
        # img_slice_7T = _get_slice_and_convert_to_tensor(img_7T, offset + slice_idx)


        vessel_3T_path = os.path.join(self.root, '3T_vessels', file_name + '.nii.gz')
        _, _, _, vessel_3T = _ants_img_info(vessel_3T_path)
        # slices_3T_vessel = self.transform_upsample(_get_all_slices(vessel_3T))
        slices_3T_vessel = _get_all_slices(vessel_3T)
        vessel_7T_path = os.path.join(self.root, '7T_vessels_downsample', file_name + '.nii.gz')
        _, _, _, vessel_7T = _ants_img_info(vessel_7T_path)
        slices_7T_vessel = _get_all_slices(vessel_7T)
        mips_path_7T = os.path.join(self.root, '7T_downsample_normal_mips', file_name + '.nii.gz')
        _, _, _, mips_7T = _ants_img_info(mips_path_7T)
        mips_slices_7T = _get_all_slices(mips_7T)  #

       
        return {'HR': img_slices_7T, 'SR': img_slices_3T, 'Index': idx, '3T vessels': slices_3T_vessel,
                '7T vessels': slices_7T_vessel, '7T mips': mips_slices_7T}

    def __len__(self):
        """Return the total number of images."""
        return 157

    @property
    def name(self):
        return self._name

    @property
    def has_labels(self):
        return False

    @property
    def has_onehot_labels(self):
        return False

    @property
    def resolution(self):
        return self._resolution

    @property
    def num_channels(self):
        return 3





class MD301TestDataset(torch.utils.data.Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, dataroot, subname, **kwargs):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        self._name = 'MD301Dataset'

        self.root = dataroot
        self.sub_name = subname
        self._resolution = 1024 

        file_list = pd.read_csv(os.path.join(dataroot, 'file_list.csv'))
        #file_list = pd.read_csv(os.path.join(dataroot, 'file_list_paired.csv'))
        self.file_list = file_list.loc[file_list['IDs'] == self.sub_name] 
        self.transform_upsample = Resize([896, 896])


    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """

        file_name = self.file_list.iloc[0]['IDs']
        # print(idx)

        img_path_3T = os.path.join(self.root, '3T_normalization', file_name + '.nii.gz')  
        _, _, _, img_3T = _ants_img_info(img_path_3T)
        img_slice_3T_1 = self.transform_upsample(_get_slice_and_convert_to_tensor(img_3T, idx))
        img_slice_3T_2 = self.transform_upsample(_get_slice_and_convert_to_tensor(img_3T, idx + 1))
        img_slice_3T_3 = self.transform_upsample(_get_slice_and_convert_to_tensor(img_3T, idx + 2))


        # img_path_7T = os.path.join(self.root, 'reno_7T', file_name + '.nii.gz')
        # _, _, _, img_7T = _ants_img_info(img_path_7T)
        # img_slices_7T = _get_all_slices(img_7T)

       
        vessel_3T_path = os.path.join(self.root, '3T_vessels', file_name + '.nii.gz')
        _, _, _, vessel_3T = _ants_img_info(vessel_3T_path)
        slices_3T_vessel_1 = self.transform_upsample(_get_slice_and_convert_to_tensor(vessel_3T, idx))
        slices_3T_vessel_2 = self.transform_upsample(_get_slice_and_convert_to_tensor(vessel_3T, idx + 1))
        slices_3T_vessel_3 = self.transform_upsample(_get_slice_and_convert_to_tensor(vessel_3T, idx + 2))



        return {'SR_1': img_slice_3T_1, 'SR_2': img_slice_3T_2, 'SR_3': img_slice_3T_3, 'SR_vessel_1': slices_3T_vessel_1,
                'SR_vessel_2': slices_3T_vessel_2, 'SR_vessel_3': slices_3T_vessel_3, 'Index': idx}

    def __len__(self):
        """Return the total number of images."""
        return 78

    @property
    def name(self):
        return self._name

    @property
    def has_labels(self):
        return False

    @property
    def has_onehot_labels(self):
        return False

    @property
    def resolution(self):
        return self._resolution

    @property
    def num_channels(self):
        return 1

