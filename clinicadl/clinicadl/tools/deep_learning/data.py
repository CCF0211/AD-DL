# coding: utf8

import torch
import pandas as pd
import numpy as np
from os import path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import abc
from clinicadl.tools.inputs.filename_types import FILENAME_TYPE
import os
import nibabel as nib
import torch.nn.functional as F
from scipy import ndimage
import socket
from utils import get_dynamic_image
from .batchgenerators.transforms.color_transforms import ContrastAugmentationTransform, BrightnessTransform, \
    GammaTransform, BrightnessGradientAdditiveTransform, LocalSmoothingTransform
from .batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform, \
    RandomShiftTransform
from .batchgenerators.transforms.noise_transforms import RicianNoiseTransform, GaussianNoiseTransform, \
    GaussianBlurTransform
from .batchgenerators.transforms.spatial_transforms import Rot90Transform, MirrorTransform, SpatialTransform
from .batchgenerators.transforms.abstract_transforms import Compose
from .batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from .data_tool import hilbert_2dto3d_cut, hilbert_3dto2d_cut, hilbert_2dto3d, hilbert_3dto2d, linear_2dto3d_cut, \
    linear_3dto2d_cut, linear_2dto3d, linear_3dto2d


#################################
# Datasets loaders
#################################


class MRIDataset(Dataset):
    """Abstract class for all derived MRIDatasets."""

    def __init__(self, caps_directory, data_file,
                 preprocessing, transformations=None):
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {
            'CN': 0,
            'AD': 1,
            'sMCI': 0,
            'pMCI': 1,
            'MCI': 2,
            'unlabeled': -1}
        self.preprocessing = preprocessing
        self.num_fake_mri = 0

        if not hasattr(self, 'elem_index'):
            raise ValueError(
                "Child class of MRIDataset must set elem_index attribute.")
        if not hasattr(self, 'mode'):
            raise ValueError(
                "Child class of MRIDataset must set mode attribute.")

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument data_file is not of correct type.')

        mandatory_col = {"participant_id", "session_id", "diagnosis"}
        if self.elem_index == "mixed":
            mandatory_col.add("%s_id" % self.mode)

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include %s" % mandatory_col)

        self.elem_per_image = self.num_elem_per_image()

    def __len__(self):
        return len(self.df) * self.elem_per_image

    def _get_path(self, participant, session, mode="image", fake_caps_path=None):

        if self.preprocessing == "t1-linear":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_linear',
                                   participant + '_' + session
                                   + FILENAME_TYPE['cropped'] + '.pt')
            origin_nii_path = path.join(self.caps_directory, 'subjects', participant, session,
                                        't1_linear', participant + '_' + session
                                        + FILENAME_TYPE['cropped'] + '.nii.gz')
            # temp_path = path.join(self.caps_directory, 'subjects', participant, session,
            #                       't1_linear')
            # for file in os.listdir(temp_path):
            #     if file.find('_run-01_') != '-1':
            #         new_name = file.replace('_run-01_', '_')
            #         os.rename(os.path.join(temp_path, file), os.path.join(temp_path, new_name))
            #         print('rename {} to {}'.format(os.path.join(temp_path, file), os.path.join(temp_path, new_name)))

            if fake_caps_path is not None:
                fake_image_path = path.join(fake_caps_path, 'subjects', participant, session,
                                            'deeplearning_prepare_data', '%s_based' % mode, 't1_spm',
                                            participant + '_' + session
                                            + FILENAME_TYPE['cropped'] + '.pt')
                fake_nii_path = path.join(fake_caps_path, 'subjects', participant, session,
                                          't1_linear', participant + '_' + session
                                          + FILENAME_TYPE['cropped'] + '.nii.gz')

                # first use fake image, because some image lacked in tsv but have in caps
                if os.path.exists(fake_image_path):
                    image_path = fake_image_path
                    self.num_fake_mri = self.num_fake_mri + 1
                elif os.path.exists(fake_nii_path):
                    image_array = nib.load(fake_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(fake_caps_path, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), fake_image_path)
                    print('save fake image: {}'.format(fake_image_path))
                    self.num_fake_mri = self.num_fake_mri + 1
                    image_path = fake_image_path
                elif os.path.exists(image_path):  # exist real pt file
                    None
                elif os.path.exists(origin_nii_path):  # exist real nii file
                    image_array = nib.load(origin_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    print(
                        'Can not find:{} and {} and {} in both real and fake folder'.format(image_path, fake_image_path,
                                                                                            fake_nii_path))

            else:
                if os.path.exists(image_path):  # exist real pt file
                    None
                elif os.path.exists(origin_nii_path):  # exist real pt file
                    image_array = nib.load(origin_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_linear')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    print('Can not find:{}'.format(image_path))
        elif self.preprocessing == "t1-extensive":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_extensive',
                                   participant + '_' + session
                                   + FILENAME_TYPE['skull_stripped'] + '.pt')
        elif self.preprocessing == "t1-spm-graymatter":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_spm',
                                   participant + '_' + session
                                   + FILENAME_TYPE['segm-graymatter'] + '.pt')
            origin_nii_path = path.join(self.caps_directory, 'subjects', participant, session,
                                        't1', 'spm', 'segmentation', 'normalized_space', participant + '_' + session
                                        + FILENAME_TYPE['segm-graymatter'] + '.nii.gz')
            temp_path = path.join(self.caps_directory, 'subjects', participant, session,
                                  't1', 'spm', 'segmentation', 'normalized_space')
            # for file in os.listdir(temp_path):
            #     if file.find('_run-01_') != '-1':
            #         new_name = file.replace('_run-01_', '_')
            #         os.rename(os.path.join(temp_path, file), os.path.join(temp_path, new_name))
            #         print('rename {} to {}'.format(os.path.join(temp_path, file), os.path.join(temp_path, new_name)))
            if fake_caps_path is not None:
                fake_image_path = path.join(fake_caps_path, 'subjects', participant, session,
                                            'deeplearning_prepare_data', '%s_based' % mode, 't1_spm',
                                            participant + '_' + session
                                            + FILENAME_TYPE['segm-graymatter'] + '.pt')
                fake_nii_path = path.join(fake_caps_path, 'subjects', participant, session,
                                          't1', 'spm', 'segmentation', 'normalized_space', participant + '_' + session
                                          + FILENAME_TYPE['segm-graymatter'] + '.nii.gz')

                # first use fake image, because some image lacked in tsv but have in caps
                if os.path.exists(fake_image_path):
                    image_path = fake_image_path
                    self.num_fake_mri = self.num_fake_mri + 1
                elif os.path.exists(fake_nii_path):
                    image_array = nib.load(fake_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(fake_caps_path, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), fake_image_path)
                    print('save fake image: {}'.format(fake_image_path))
                    self.num_fake_mri = self.num_fake_mri + 1
                    image_path = fake_image_path
                elif os.path.exists(image_path):  # exist real pt file
                    None
                elif os.path.exists(origin_nii_path):  # exist real pt file
                    image_array = nib.load(origin_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    print(
                        'Can not find:{} and {} and {} in both real and fake folder'.format(image_path, fake_image_path,
                                                                                            fake_nii_path))

            else:
                if os.path.exists(image_path):  # exist real pt file
                    None
                elif os.path.exists(origin_nii_path):  # exist real pt file
                    image_array = nib.load(origin_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    print('Can not find:{}'.format(image_path))
                    print('Can not find:{}'.format(origin_nii_path))

        elif self.preprocessing == "t1-spm-whitematter":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_spm',
                                   participant + '_' + session
                                   + FILENAME_TYPE['segm-whitematter'] + '.pt')
            origin_nii_path = path.join(self.caps_directory, 'subjects', participant, session,
                                        't1', 'spm', 'segmentation', 'normalized_space', participant + '_' + session
                                        + FILENAME_TYPE['segm-whitematter'] + '.nii.gz')
            if fake_caps_path is not None:
                fake_image_path = path.join(fake_caps_path, 'subjects', participant, session,
                                            'deeplearning_prepare_data', '%s_based' % mode, 't1_spm',
                                            participant + '_' + session
                                            + FILENAME_TYPE['segm-whitematter'] + '.pt')
                fake_nii_path = path.join(fake_caps_path, 'subjects', participant, session,
                                          't1', 'spm', 'segmentation', 'normalized_space', participant + '_' + session
                                          + FILENAME_TYPE['segm-whitematter'] + '.nii.gz')

                # first use fake image, because some image lacked in tsv but have in caps
                if os.path.exists(fake_image_path):
                    image_path = fake_image_path
                    self.num_fake_mri = self.num_fake_mri + 1
                elif os.path.exists(fake_nii_path):
                    image_array = nib.load(fake_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(fake_caps_path, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), fake_image_path)
                    image_path = fake_image_path
                    print('save fake image: {}'.format(fake_image_path))
                    self.num_fake_mri = self.num_fake_mri + 1
                elif os.path.exists(image_path):  # exist real pt file
                    None
                elif os.path.exists(origin_nii_path):  # exist real pt file
                    image_array = nib.load(origin_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    print('Can not find:{}'.format(image_path))

            else:

                if os.path.exists(image_path):  # exist real pt file
                    None
                elif os.path.exists(origin_nii_path):  # exist real pt file
                    image_array = nib.load(origin_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    print('Can not find:{}'.format(image_path))
        elif self.preprocessing == "t1-spm-csf":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_spm',
                                   participant + '_' + session
                                   + FILENAME_TYPE['segm-csf'] + '.pt')
            origin_nii_path = path.join(self.caps_directory, 'subjects', participant, session,
                                        't1', 'spm', 'segmentation', 'normalized_space', participant + '_' + session
                                        + FILENAME_TYPE['segm-csf'] + '.nii.gz')
            if fake_caps_path is not None:
                fake_image_path = path.join(fake_caps_path, 'subjects', participant, session,
                                            'deeplearning_prepare_data', '%s_based' % mode, 't1_spm',
                                            participant + '_' + session
                                            + FILENAME_TYPE['segm-csf'] + '.pt')
                fake_nii_path = path.join(fake_caps_path, 'subjects', participant, session,
                                          't1', 'spm', 'segmentation', 'normalized_space', participant + '_' + session
                                          + FILENAME_TYPE['segm-csf'] + '.nii.gz')

                # first use fake image, because some image lacked in tsv but have in caps
                if os.path.exists(fake_image_path):
                    image_path = fake_image_path
                    self.num_fake_mri = self.num_fake_mri + 1
                elif os.path.exists(fake_nii_path):
                    image_array = nib.load(fake_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(fake_caps_path, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), fake_image_path)
                    image_path = fake_image_path
                    print('save fake image: {}'.format(fake_image_path))
                    self.num_fake_mri = self.num_fake_mri + 1
                elif os.path.exists(image_path):  # exist real pt file
                    None
                elif os.path.exists(origin_nii_path):  # exist real pt file
                    image_array = nib.load(origin_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    print('Can not find:{}'.format(image_path))

            else:

                if os.path.exists(image_path):  # exist real pt file
                    None
                elif os.path.exists(origin_nii_path):  # exist real pt file
                    image_array = nib.load(origin_nii_path).get_fdata()
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                    save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                         'deeplearning_prepare_data', '%s_based' % mode, 't1_spm')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    print('Can not find:{}'.format(image_path))

        return image_path

    def _get_meta_data(self, idx):
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, 'participant_id']
        session = self.df.loc[image_idx, 'session_id']

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        elif self.elem_index == "mixed":
            elem_idx = self.df.loc[image_idx, '%s_id' % self.mode]
        else:
            elem_idx = self.elem_index

        diagnosis = self.df.loc[image_idx, 'diagnosis']
        label = self.diagnosis_code[diagnosis]

        return participant, session, elem_idx, label

    def _get_full_image(self):
        from ..data.utils import find_image_path as get_nii_path
        import nibabel as nib

        if self.preprocessing in ["t1-linear", "t1-extensive"]:
            participant_id = self.df.loc[0, 'participant_id']
            session_id = self.df.loc[0, 'session_id']
            try:
                image_path = self._get_path(participant_id, session_id, "image")
                image = torch.load(image_path)
            except FileNotFoundError:
                try:
                    image_path = get_nii_path(
                        self.caps_directory,
                        participant_id,
                        session_id,
                        preprocessing=self.preprocessing)
                    image_nii = nib.load(image_path)
                    image_np = image_nii.get_fdata()
                    image = ToTensor()(image_np)
                except:
                    # if we use moved folder which only has slice/patch, we can not find the whole image in folder, so use this file to get full image
                    # image_path = os.path.join(self.caps_directory,'sub-ADNI002S0295_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz')
                    # image_nii = nib.load(image_path)
                    # image_np = image_nii.get_fdata()
                    # image = ToTensor()(image_np)
                    image = torch.zeros([169, 208, 179])  # in those segm data, size : [169, 208, 179]
        elif self.preprocessing in ["t1-spm-whitematter", "t1-spm-whitematter", "t1-spm-csf"]:
            image = torch.zeros([121, 145, 121])  # in those segm data, size : [121, 145, 121]

        return image

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def num_elem_per_image(self):
        pass


class MRIDatasetImage(MRIDataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, caps_directory, data_file,
                 preprocessing='t1-linear', transformations=None, crop_padding_to_128=False, resample_size=None,
                 fake_caps_path=None, roi=False, roi_size=32, model=None, data_preprocess='MinMax',
                 data_Augmentation=False, method_2d=None):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.

        """
        self.elem_index = None
        self.mode = "image"
        self.model = model
        self.data_preprocess = data_preprocess
        self.data_Augmentation = data_Augmentation
        self.crop_padding_to_128 = crop_padding_to_128
        self.resample_size = resample_size
        self.fake_caps_path = fake_caps_path
        self.roi = roi
        self.roi_size = roi_size
        self.method_2d = method_2d
        # if self.roi:
        #     if socket.gethostname() == 'zkyd':
        #         aal_mask_dict_dir = '/root/Downloads/atlas/aal_mask_dict_128.npy'
        #     elif socket.gethostname() == 'tian-W320-G10':
        #         aal_mask_dict_dir = '/home/tian/pycharm_project/MRI_GNN/atlas/aal_mask_dict_128.npy'
        #     self.aal_mask_dict = np.load(aal_mask_dict_dir, allow_pickle=True).item()  # 116; (181,217,181)
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        print('crop_padding_to_128 type:{}'.format(self.crop_padding_to_128))

    def __getitem__(self, idx):
        participant, session, _, label = self._get_meta_data(idx)
        image_path = self._get_path(participant, session, "image", fake_caps_path=self.fake_caps_path)

        if self.preprocessing == 't1-linear':
            ori_name = 't1_linear'
        else:
            ori_name = 't1_spm'
        resampled_image_path = image_path.replace(ori_name, '{}_{}_resample_{}'.format(ori_name, self.data_preprocess,
                                                                                       self.resample_size))
        CNN2020_DEEPCNN_image_path = image_path.replace(ori_name,
                                                        '{}_{}_model_{}'.format(ori_name, self.data_preprocess,
                                                                                self.model))

        roi_image_path = resampled_image_path.replace('image_based',
                                                      'AAL_roi_based_{}'.format(self.roi_size))
        # delate_image_path = image_path.replace('image_based',
        #                                        'AAL_roi_based_{}'.format(self.roi_size))
        # if os.path.exists(delate_image_path):
        #     os.remove(delate_image_path)
        #     print('delating:{}'.format(delate_image_path))
        if not self.data_Augmentation:  # No data_Augmentation, 1. check local disk whether have saved data. 2. If not, process data and save to desk
            if self.roi and 'ROI' in self.model:
                # Get resampled_image
                if os.path.exists(resampled_image_path):
                    try:
                        resampled_image = torch.load(resampled_image_path)
                        # print('loading:{}'.format(roi_image_path))
                    except:
                        print('Wrong file:{}'.format(resampled_image_path))
                else:
                    image = torch.load(image_path)
                    if self.transformations:
                        dict = {}
                        dict['data'] = image
                        resampled_image = self.transformations(begin_trans_indx=0, **dict)
                    resampled_data = resampled_image.squeeze()  # [128, 128, 128]
                    try:
                        resampled_image = resampled_data.unsqueeze(dim=0)  # [1, 128, 128, 128]
                    except:
                        resampled_image = np.expand_dims(resampled_data, 0)
                    dir, file = os.path.split(resampled_image_path)
                    if not os.path.exists(dir):
                        try:
                            os.makedirs(dir)
                        except OSError:
                            pass
                    torch.save(resampled_image, resampled_image_path)
                    print('Save resampled {} image: {}'.format(self.resample_size, resampled_image_path))
                # Get roi image
                if os.path.exists(roi_image_path):
                    try:
                        ROI_image = torch.load(roi_image_path)
                        # print('loading:{}'.format(roi_image_path))
                    except:
                        print('Wrong file:{}'.format(roi_image_path))
                else:
                    image = torch.load(image_path)
                    if self.transformations:
                        dict = {}
                        dict['data'] = image
                        image = self.transformations(begin_trans_indx=0, **dict)
                    data = image.squeeze()  # [128, 128, 128]
                    data = self.roi_extract(data, roi_size=self.roi_size, sub_id=participant,
                                            preprocessing=self.preprocessing,
                                            session=session, save_nii=False)
                    ROI_image = data.unsqueeze(dim=0)  # [1, num_roi, 128, 128, 128]

                    # sample = {'image': image, 'roi_image': ROI_image, 'label': label, 'participant_id': participant,
                    #           'session_id': session,
                    #           'image_path': image_path, 'num_fake_mri': self.num_fake_mri}
                    dir, file = os.path.split(roi_image_path)
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    torch.save(ROI_image, roi_image_path)
                    print('Save roi image: {}'.format(roi_image_path))

                sample = {'image': ROI_image, 'label': label, 'participant_id': participant,
                          'session_id': session, 'all_image': resampled_image,
                          'image_path': roi_image_path, 'num_fake_mri': self.num_fake_mri}
            elif self.model in ["CNN2020", "DeepCNN"]:
                if os.path.exists(CNN2020_DEEPCNN_image_path):
                    CNN2020_DEEPCNN_image_image = torch.load(CNN2020_DEEPCNN_image_path)
                else:
                    image = torch.load(image_path)
                    if self.transformations:
                        dict = {}
                        dict['data'] = image
                        image = self.transformations(begin_trans_indx=0, **dict)
                    data = image.squeeze()  # [128, 128, 128]
                    try:
                        CNN2020_DEEPCNN_image_image = data.unsqueeze(dim=0)  # [1, 128, 128, 128]
                    except:
                        CNN2020_DEEPCNN_image_image = np.expand_dims(data, 0)
                    dir, file = os.path.split(CNN2020_DEEPCNN_image_path)
                    if not os.path.exists(dir):
                        try:
                            os.makedirs(dir)
                        except OSError:
                            pass
                    torch.save(image, CNN2020_DEEPCNN_image_path)
                    print('Save resampled {} image: {}'.format(self.resample_size, CNN2020_DEEPCNN_image_path))
                sample = {'image': CNN2020_DEEPCNN_image_image, 'label': label, 'participant_id': participant,
                          'session_id': session,
                          'image_path': CNN2020_DEEPCNN_image_path, 'num_fake_mri': self.num_fake_mri}
            elif self.method_2d is not None:
                path, file = os.path.split(resampled_image_path)
                file_2d = file.split('.')[0] + '_' + self.method_2d + '.' + file.split('.')[1]
                path_2d = os.path.join(path, file_2d)
                if os.path.exists(path_2d):
                    try:
                        data_2d = torch.load(path_2d)
                    except:
                        print('Wrong file:{}'.format(path_2d))
                else:
                    if os.path.exists(resampled_image_path):
                        try:
                            resampled_image = torch.load(resampled_image_path)
                            # print('loading:{}'.format(roi_image_path))
                        except:
                            print('Wrong file:{}'.format(resampled_image_path))
                    else:
                        image = torch.load(image_path)
                        if self.transformations:
                            dict = {}
                            dict['data'] = image
                            resampled_image = self.transformations(begin_trans_indx=0, **dict)
                        resampled_data = resampled_image.squeeze()  # [128, 128, 128]
                        try:
                            resampled_image = resampled_data.unsqueeze(dim=0)  # [1, 128, 128, 128]
                        except:
                            resampled_image = np.expand_dims(resampled_data, 0)
                        dir, file = os.path.split(resampled_image_path)
                        if not os.path.exists(dir):
                            try:
                                os.makedirs(dir)
                            except OSError:
                                pass
                        torch.save(resampled_image, resampled_image_path)
                        print('Save resampled {} image: {}'.format(self.resample_size, resampled_image_path))
                    if self.method_2d == 'hilbert_cut':
                        data_2d = hilbert_3dto2d_cut(resampled_image)
                        torch.save(data_2d.clone(), path_2d)
                        print('saving:{}'.format(path_2d))
                    elif self.method_2d == 'linear_cut':
                        data_2d = linear_3dto2d_cut(resampled_image)
                        torch.save(data_2d.clone(), path_2d)
                        print('saving:{}'.format(path_2d))
                    elif self.method_2d == 'hilbert_downsampling':
                        data_low = self.__resize_data__(resampled_image.squeeze(), target_size=[64, 64, 64])
                        data_low = torch.from_numpy(data_low)
                        data_2d = hilbert_3dto2d(data_low)
                        data_2d = data_2d.unsqueeze(0)  # [1,512,512]
                        torch.save(data_2d.clone(), path_2d)
                        print('saving:{}'.format(path_2d))
                    elif self.method_2d == 'linear_downsampling':
                        data_low = self.__resize_data__(resampled_image.squeeze(), target_size=[64, 64, 64])
                        data_low = torch.from_numpy(data_low)
                        data_2d = linear_3dto2d(data_low)
                        data_2d = data_2d.unsqueeze(0)  # [1,512,512]
                        torch.save(data_2d.clone(), path_2d)
                        print('saving:{}'.format(path_2d))
                sample = {'image': data_2d.squeeze(), 'label': label, 'participant_id': participant,
                          'session_id': session,
                          'image_path': path_2d, 'num_fake_mri': self.num_fake_mri}
            elif self.model not in [
                "Conv5_FC3",
                'DeepCNN',
                'CNN2020',
                'CNN2020_gcn',
                'DeepCNN_gcn',
                "Dynamic2D_net_Alex",
                "Dynamic2D_net_Res34",
                "Dynamic2D_net_Res18",
                "Dynamic2D_net_Vgg16",
                "Dynamic2D_net_Vgg11",
                "Dynamic2D_net_Mobile",
                'ROI_GCN']:
                if os.path.exists(resampled_image_path):
                    try:
                        resampled_image = torch.load(resampled_image_path)
                    except:
                        raise FileExistsError('file error:{}'.format(resampled_image_path))
                    # if self.data_Augmentation and self.transformations:
                    #     dict = {}
                    #     dict['data'] = resampled_image
                    #     begin_trans_indx = 0
                    #     for i in range(len(self.transformations.transforms)):
                    #         if self.transformations.transforms[i].__class__.__name__ in ['ItensityNormalizeNonzeorVolume',
                    #                                                                      'ItensityNormalizeNonzeorVolume',
                    #                                                                      'MinMaxNormalization']:
                    #             begin_trans_indx = i + 1
                    #     resampled_image = self.transformations(begin_trans_indx=begin_trans_indx, **dict)
                else:
                    image = torch.load(image_path)
                    if self.transformations:
                        dict = {}
                        dict['data'] = image
                        resampled_image = self.transformations(begin_trans_indx=0, **dict)
                    resampled_data = resampled_image.squeeze()  # [128, 128, 128]
                    try:
                        resampled_image = resampled_data.unsqueeze(dim=0)  # [1, 128, 128, 128]
                    except:
                        resampled_image = np.expand_dims(resampled_data, 0)
                    dir, file = os.path.split(resampled_image_path)
                    if not os.path.exists(dir):
                        try:
                            os.makedirs(dir)
                        except OSError:
                            pass
                    torch.save(image, resampled_image_path)
                    print('Save resampled {} image: {}'.format(self.resample_size, resampled_image_path))

                sample = {'image': resampled_image, 'label': label, 'participant_id': participant,
                          'session_id': session,
                          'image_path': resampled_image_path, 'num_fake_mri': self.num_fake_mri}
            elif self.model in ["Dynamic2D_net_Alex", "Dynamic2D_net_Res34", "Dynamic2D_net_Res18",
                                "Dynamic2D_net_Vgg16", "Dynamic2D_net_Vgg11", "Dynamic2D_net_Mobile"]:
                image = torch.load(image_path)
                if self.transformations:
                    dict = {}
                    dict['data'] = image
                    resampled_image = self.transformations(begin_trans_indx=0, **dict)
                resampled_data = resampled_image.squeeze()  # [128, 128, 128]
                image_np = np.array(resampled_data)
                image_np = np.expand_dims(image_np, 0)  # 0,w,h,d
                image_np = np.swapaxes(image_np, 0, 3)  # w,h,d,0
                im = get_dynamic_image(image_np)
                im = np.expand_dims(im, 0)
                im = np.concatenate([im, im, im], 0)
                im = torch.from_numpy(im)
                im = im.float()
                sample = {'image': im, 'label': label, 'participant_id': participant, 'session_id': session,
                          'image_path': image_path, 'num_fake_mri': self.num_fake_mri}
                return sample
            return sample
        else:  # Use data_Augmentation, 1. Just load original data and process it
            # 1. load original data
            image = torch.load(image_path)
            if self.transformations:  # Augmentation
                dict = {}
                dict['data'] = image
                image = self.transformations(begin_trans_indx=0, **dict)
            augmentation_data = image.squeeze()  # [128, 128, 128]
            # print(self.transformations)
            # print(self.transformations[0])
            # if self.crop_padding_to_128 and image.shape[1] != 128:
            #     image = image[:, :, 8:-9, :]  # [1, 121, 128, 121]
            #     image = image.unsqueeze(0)  # [1, 1, 121, 128, 121]
            #     pad = torch.nn.ReplicationPad3d((4, 3, 0, 0, 4, 3))
            #     image = pad(image)  # [1, 1, 128, 128, 128]
            #     image = image.squeeze(0)  # [1, 128, 128, 128]
            # if self.resample_size is not None:
            #     assert self.resample_size > 0, 'resample_size should be a int positive number'
            #     image = image.unsqueeze(0)
            #     image = F.interpolate(image,
            #                           size=self.resample_size)  # resize to resample_size * resample_size * resample_size
            #     print('resample before trans shape:{}'.format(image.shape))
            #     print('resample before trans mean:{}'.format(image.mean()))
            #     print('resample before trans std:{}'.format(image.std()))
            #     print('resample before trans max:{}'.format(image.max()))
            #     print('resample before trans min:{}'.format(image.min()))
            #     # image = self.transformations(image)
            #     # print('resample after trans shape:{}'.format(image.shape))
            #     # print('resample after trans mean:{}'.format(image.mean()))
            #     # print('resample after trans std:{}'.format(image.std()))
            #     # print('resample after trans max:{}'.format(image.max()))
            #     # print('resample after trans min:{}'.format(image.min()))
            #     image = image.squeeze(0)
            #
            # if self.model in ['DeepCNN', 'DeepCNN_gcn']:
            #     image = image.unsqueeze(0)
            #     image = F.interpolate(image, size=[49, 39, 38])
            #     image = image.squeeze(0)
            # elif self.model in ['CNN2020', 'CNN2020_gcn']:
            #     image = image.unsqueeze(0)
            #     image = F.interpolate(image, size=[139, 177, 144])
            #     image = image.squeeze(0)
            # # preprocessing data
            # data = image.squeeze()  # [128, 128, 128]
            # # print(data.shape)
            # input_W, input_H, input_D = data.shape
            # if self.model not in ["ConvNet3D", "ConvNet3D_gcn", "VoxCNN", "Conv5_FC3", 'DeepCNN', 'CNN2020', 'CNN2020_gcn',
            #                       "VoxCNN_gcn", 'DeepCNN_gcn', "ConvNet3D_v2", "ConvNet3D_ori", "Dynamic2D_net_Alex",
            #                       "Dynamic2D_net_Res34", "Dynamic2D_net_Res18", "Dynamic2D_net_Vgg16",
            #                       "Dynamic2D_net_Vgg11", "Dynamic2D_net_Mobile"]:
            #     # drop out the invalid range
            #     # if self.preprocessing in ['t1-spm-graymatter', 't1-spm-whitematter', 't1-spm-csf']:
            #     data = self.__drop_invalid_range__(data)
            #     print('drop_invalid_range shape:{}'.format(data.shape))
            #     print('drop_invalid_range mean:{}'.format(data.mean()))
            #     print('drop_invalid_range std:{}'.format(data.std()))
            #     print('drop_invalid_range max:{}'.format(data.max()))
            #     print('drop_invalid_range min:{}'.format(data.min()))
            #     # resize data
            #     data = self.__resize_data__(data, input_W, input_H, input_D)
            #     print('resize_data shape:{}'.format(data.shape))
            #     print('resize_data mean:{}'.format(data.mean()))
            #     print('resize_data std:{}'.format(data.std()))
            #     print('resize_data max:{}'.format(data.max()))
            #     print('resize_data min:{}'.format(data.min()))
            #     # normalization datas
            #     data = np.array(data)
            #     data = self.__itensity_normalize_one_volume__(data)
            #     print('itensity_normalize shape:{}'.format(data.shape))
            #     print('itensity_normalize mean:{}'.format(data.mean()))
            #     print('itensity_normalize std:{}'.format(data.std()))
            #     print('itensity_normalize max:{}'.format(data.max()))
            #     print('itensity_normalize min:{}'.format(data.min()))
            #     # if self.transformations and self.model in ["ConvNet3D", "VoxCNN"]:
            #     #     data = self.transformations(data)
            #     data = torch.from_numpy(data)
            # if self.model in ['CNN2020', 'CNN2020_gcn']:
            #     data = np.array(data)
            #     data = self.__itensity_normalize_one_volume__(data, normalize_all=True)
            #     data = torch.from_numpy(data)
            if self.model in ["Dynamic2D_net_Alex", "Dynamic2D_net_Res34", "Dynamic2D_net_Res18",
                              "Dynamic2D_net_Vgg16", "Dynamic2D_net_Vgg11", "Dynamic2D_net_Mobile"]:
                image_np = np.array(augmentation_data)
                image_np = np.expand_dims(image_np, 0)  # 0,w,h,d
                image_np = np.swapaxes(image_np, 0, 3)  # w,h,d,0
                im = get_dynamic_image(image_np)
                im = np.expand_dims(im, 0)
                im = np.concatenate([im, im, im], 0)
                im = torch.from_numpy(im)
                im = im.float()
                sample = {'image': im, 'label': label, 'participant_id': participant, 'session_id': session,
                          'image_path': image_path, 'num_fake_mri': self.num_fake_mri}
                return sample

            if self.roi and 'ROI' in self.model:
                try:
                    resampled_image = augmentation_data.unsqueeze(dim=0)  # [1, 128, 128, 128]
                except:
                    resampled_image = np.expand_dims(augmentation_data, 0)
                augmentation_data = self.roi_extract(augmentation_data, roi_size=self.roi_size, sub_id=participant,
                                                     preprocessing=self.preprocessing,
                                                     session=session, save_nii=False)
                ROI_image = augmentation_data.unsqueeze(dim=0)  # [1, num_roi, 128, 128, 128]
                sample = {'image': ROI_image, 'all_image': resampled_image, 'label': label,
                          'participant_id': participant,
                          'session_id': session,
                          'image_path': image_path, 'num_fake_mri': self.num_fake_mri}
            elif self.method_2d is not None:
                path, file = os.path.split(resampled_image_path)
                file_2d = file.split('.')[0] + '_' + self.method_2d + '.' + file.split('.')[1]
                path_2d = os.path.join(path, file_2d)
                try:
                    resampled_image = augmentation_data.unsqueeze(dim=0)  # [1, 128, 128, 128]
                except:
                    resampled_image = np.expand_dims(augmentation_data, 0)
                if self.method_2d == 'hilbert_cut':
                    data_2d = hilbert_3dto2d_cut(resampled_image)
                elif self.method_2d == 'linear_cut':
                    data_2d = linear_3dto2d_cut(resampled_image)
                elif self.method_2d == 'hilbert_downsampling':
                    data_low = self.__resize_data__(resampled_image.squeeze(), target_size=[64, 64, 64])
                    data_low = torch.from_numpy(data_low)
                    data_2d = hilbert_3dto2d(data_low)
                    data_2d = data_2d.unsqueeze(0)  # [1,512,512]
                elif self.method_2d == 'linear_downsampling':
                    data_low = self.__resize_data__(resampled_image.squeeze(), target_size=[64, 64, 64])
                    data_low = torch.from_numpy(data_low)
                    data_2d = linear_3dto2d(data_low)
                    data_2d = data_2d.unsqueeze(0)  # [1,512,512]
                sample = {'image': data_2d.squeeze(), 'label': label, 'participant_id': participant,
                          'session_id': session,
                          'image_path': path_2d, 'num_fake_mri': self.num_fake_mri}
            elif self.model in ["CNN2020", "DeepCNN"]:
                try:
                    CNN2020_DEEPCNN_image_image = augmentation_data.unsqueeze(dim=0)  # [1, 128, 128, 128]
                except:
                    CNN2020_DEEPCNN_image_image = np.expand_dims(augmentation_data, 0)
                dir, file = os.path.split(CNN2020_DEEPCNN_image_path)
                if not os.path.exists(dir):
                    try:
                        os.makedirs(dir)
                    except OSError:
                        pass
                torch.save(image, CNN2020_DEEPCNN_image_path)
                print('Save resampled {} image: {}'.format(self.resample_size, CNN2020_DEEPCNN_image_path))
                sample = {'image': CNN2020_DEEPCNN_image_image, 'label': label, 'participant_id': participant,
                          'session_id': session,
                          'image_path': CNN2020_DEEPCNN_image_path, 'num_fake_mri': self.num_fake_mri}

            else:
                try:
                    image = augmentation_data.unsqueeze(dim=0)  # [1, 128, 128, 128]
                except:
                    image = np.expand_dims(augmentation_data, 0)

                sample = {'image': image, 'label': label, 'participant_id': participant, 'session_id': session,
                          'image_path': image_path, 'num_fake_mri': self.num_fake_mri}

            return sample

    def __drop_invalid_range__(self, volume):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        # print('zero:{}'.format(zero_value))
        non_zeros_idx = np.where(volume != zero_value)
        # print('zero idx:{}'.format(non_zeros_idx))
        try:
            [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
            [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        except:
            print(zero_value)
            print(non_zeros_idx)

        return volume[min_z:max_z + 1, min_h:max_h + 1, min_w:max_w + 1]

    def __resize_data__(self, data, input_W, input_H, input_D):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [input_W * 1.0 / depth, input_H * 1.0 / height, input_D * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __itensity_normalize_one_volume__(self, volume, normalize_all=False):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        if normalize_all:
            pixels = volume
        else:
            pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        if not normalize_all:
            out_random = np.random.normal(0, 1, size=volume.shape)
            out[volume == 0] = out_random[volume == 0]
        return out

    def num_elem_per_image(self):
        return 1

    def roi_extract(self, MRI, roi_size=32, sub_id=None, preprocessing=None, session=None, save_nii=False):
        roi_data_list = []
        roi_label_list = []
        if 'slave' in socket.gethostname():
            aal_mask_dict_dir = '/root/Downloads/atlas/aal_mask_dict_right.npy'
        elif socket.gethostname() == 'tian-W320-G10':
            aal_mask_dict_dir = '/home/tian/pycharm_project/MRI_GNN/atlas/aal_mask_dict_right.npy'
        elif socket.gethostname() == 'zkyd':
            aal_mask_dict_dir = '/data/fanchenchen/atlas/aal_mask_dict_right.npy'
        self.aal_mask_dict = np.load(aal_mask_dict_dir, allow_pickle=True).item()  # 116; (181,217,181)
        for i, key in enumerate(self.aal_mask_dict.keys()):
            # useful_data = self.__drop_invalid_range__(self.aal_mask_dict[key])
            # useful_data = resize_data(useful_data, target_size=[128, 128, 128])
            # useful_data = useful_data[np.newaxis, np.newaxis, :, :, :]  # 1,1,128,128,128
            # roi_batch_data = MRI.cpu().numpy() * useful_data  # batch, 1, 128,128,128
            mask = self.aal_mask_dict[key]
            # print('mask min:{}'.format(mask.min()))
            # print('mask max:{}'.format(mask.max()))
            # print('mask:{}'.format(mask))
            ww, hh, dd = MRI.shape

            MRI = self.__resize_data__(MRI, 181, 217, 181)
            # MRI = (MRI - MRI.min()) / (MRI.max() - MRI.min())
            roi_data = MRI * mask.squeeze()  # batch, 1, 128,128,128
            # print('roi_data min:{}'.format(roi_data.min()))
            # print('roi_data max:{}'.format(roi_data.max()))
            roi_label_list.append(key)
            # save nii to Visualization
            # print(image_np.max())
            # print(image_np.min())
            # print(roi_data.shape)
            if save_nii:
                image_nii = nib.Nifti1Image(roi_data, np.eye(4))
                MRI_path = '/data/fanchenchen/atlas/{}_{}_{}_ori_roi_{}.nii.gz'.format(sub_id, session,
                                                                                       preprocessing, i)
                nib.save(image_nii, MRI_path)
            try:
                roi_data = self.__drop_invalid_range__(roi_data)  # xx,xx,xx
            except:
                print(sub_id)
                print(session)
                assert True
            # roi_data = self.__drop_invalid_range__(mask.squeeze())  # xx,xx,xx
            if save_nii:
                image_nii = nib.Nifti1Image(roi_data, np.eye(4))
                MRI_path = '/data/fanchenchen/atlas/{}_{}_{}_drop_invalid_roi_{}.nii.gz'.format(sub_id, session,
                                                                                                preprocessing, i)
                nib.save(image_nii, MRI_path)
            # print(roi_data.shape)
            roi_data = self.__resize_data__(roi_data, roi_size, roi_size, roi_size)  # roi_size, roi_size, roi_size
            # print(roi_data.shape)
            roi_data = torch.from_numpy(roi_data)
            roi_data_list.append(roi_data)  # roi_size, roi_size, roi_size
            # save nii to Visualization
            if save_nii:
                image_np = roi_data.numpy()
                image_nii = nib.Nifti1Image(image_np, np.eye(4))
                MRI_path = '/data/fanchenchen/atlas/{}_{}_{}_resize_roi_{}.nii.gz'.format(sub_id, session,
                                                                                          preprocessing, i)
                nib.save(image_nii, MRI_path)
            if i >= 89:
                break

        roi_batch = torch.stack(roi_data_list).type(torch.float32)  # num_roi, roi_size, roi_size, roi_size
        return roi_batch


class MRIDatasetPatch(MRIDataset):

    def __init__(self, caps_directory, data_file, patch_size, stride_size, transformations=None, prepare_dl=False,
                 patch_index=None, preprocessing="t1-linear"):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.
            prepare_dl (bool): If true pre-extracted patches will be loaded.
            patch_index (int, optional): If a value is given the same patch location will be extracted for each image.
                else the dataset will load all the patches possible for one image.
            patch_size (int): size of the regular cubic patch.
            stride_size (int): length between the centers of two patches.

        """
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.elem_index = patch_index
        self.mode = "patch"
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

    def __getitem__(self, idx):
        participant, session, patch_idx, label = self._get_meta_data(idx)

        if self.prepare_dl:
            patch_path = path.join(self._get_path(participant, session, "patch")[0:-7]
                                   + '_patchsize-' + str(self.patch_size)
                                   + '_stride-' + str(self.stride_size)
                                   + '_patch-' + str(patch_idx) + '_T1w.pt')

            image = torch.load(patch_path)
        else:
            image_path = self._get_path(participant, session, "image")
            full_image = torch.load(image_path)
            image = self.extract_patch_from_mri(full_image, patch_idx)

        if self.transformations:
            image = self.transformations(image)

        sample = {'image': image, 'label': label,
                  'participant_id': participant, 'session_id': session, 'patch_id': patch_idx}

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1

        image = self._get_full_image()

        patches_tensor = image.unfold(1, self.patch_size, self.stride_size
                                      ).unfold(2, self.patch_size, self.stride_size
                                               ).unfold(3, self.patch_size, self.stride_size).contiguous()
        patches_tensor = patches_tensor.view(-1,
                                             self.patch_size,
                                             self.patch_size,
                                             self.patch_size)
        num_patches = patches_tensor.shape[0]
        return num_patches

    def extract_patch_from_mri(self, image_tensor, index_patch):

        patches_tensor = image_tensor.unfold(1, self.patch_size, self.stride_size
                                             ).unfold(2, self.patch_size, self.stride_size
                                                      ).unfold(3, self.patch_size, self.stride_size).contiguous()
        patches_tensor = patches_tensor.view(-1,
                                             self.patch_size,
                                             self.patch_size,
                                             self.patch_size)
        extracted_patch = patches_tensor[index_patch, ...].unsqueeze_(
            0).clone()

        return extracted_patch


class MRIDatasetRoi(MRIDataset):

    def __init__(self, caps_directory, data_file, preprocessing="t1-linear",
                 transformations=None, prepare_dl=False):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.
            prepare_dl (bool): If true pre-extracted patches will be loaded.

        """
        self.elem_index = None
        self.mode = "roi"
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

    def __getitem__(self, idx):
        participant, session, roi_idx, label = self._get_meta_data(idx)

        if self.prepare_dl:
            raise NotImplementedError(
                'The extraction of ROIs prior to training is not implemented.')

        else:
            image_path = self._get_path(participant, session, "image")
            image = torch.load(image_path)
            patch = self.extract_roi_from_mri(image, roi_idx)

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image': patch, 'label': label,
                  'participant_id': participant, 'session_id': session,
                  'roi_id': roi_idx}

        return sample

    def num_elem_per_image(self):
        return 2

    def extract_roi_from_mri(self, image_tensor, left_is_odd):
        """

        :param image_tensor: (Tensor) the tensor of the image.
        :param left_is_odd: (int) if 1 the left hippocampus is extracted, else the right one.
        :return: Tensor of the extracted hippocampus
        """

        if self.preprocessing == "t1-linear":
            if left_is_odd == 1:
                # the center of the left hippocampus
                crop_center = (61, 96, 68)
            else:
                # the center of the right hippocampus
                crop_center = (109, 96, 68)
        else:
            raise NotImplementedError("The extraction of hippocampi was not implemented for "
                                      "preprocessing %s" % self.preprocessing)
        crop_size = (50, 50, 50)  # the output cropped hippocampus size

        extracted_roi = image_tensor[
                        :,
                        crop_center[0] - crop_size[0] // 2: crop_center[0] + crop_size[0] // 2:,
                        crop_center[1] - crop_size[1] // 2: crop_center[1] + crop_size[1] // 2:,
                        crop_center[2] - crop_size[2] // 2: crop_center[2] + crop_size[2] // 2:
                        ].clone()

        return extracted_roi


class MRIDatasetSlice(MRIDataset):

    def __init__(self, caps_directory, data_file, preprocessing="t1-linear",
                 transformations=None, mri_plane=0, prepare_dl=False,
                 discarded_slices=20, mixed=False):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.
            prepare_dl (bool): If true pre-extracted patches will be loaded.
            mri_plane (int): Defines which mri plane is used for slice extraction.
            discarded_slices (int or list): number of slices discarded at the beginning and the end of the image.
                If one single value is given, the same amount is discarded at the beginning and at the end.
            mixed (bool): If True will look for a 'slice_id' column in the input DataFrame to load each slice
                independently.
        """
        # Rename MRI plane
        self.mri_plane = mri_plane
        self.direction_list = ['sag', 'cor', 'axi']
        if self.mri_plane >= len(self.direction_list):
            raise ValueError(
                "mri_plane value %i > %i" %
                (self.mri_plane, len(
                    self.direction_list)))

        # Manage discarded_slices
        if isinstance(discarded_slices, int):
            discarded_slices = [discarded_slices, discarded_slices]
        if isinstance(discarded_slices, list) and len(discarded_slices) == 1:
            discarded_slices = discarded_slices * 2
        self.discarded_slices = discarded_slices

        if mixed:
            self.elem_index = "mixed"
        else:
            self.elem_index = None

        self.mode = "slice"
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

    def __getitem__(self, idx):
        participant, session, slice_idx, label = self._get_meta_data(idx)
        slice_idx = slice_idx + self.discarded_slices[0]

        if self.prepare_dl:
            # read the slices directly
            slice_path = path.join(self._get_path(participant, session, "slice")[0:-7]
                                   + '_axis-%s' % self.direction_list[self.mri_plane]
                                   + '_channel-rgb_slice-%i_T1w.pt' % slice_idx)
            image = torch.load(slice_path)
        else:
            image_path = self._get_path(participant, session, "image")
            full_image = torch.load(image_path)
            image = self.extract_slice_from_mri(full_image, slice_idx)

        if self.transformations:
            image = self.transformations(image)

        sample = {'image': image, 'label': label,
                  'participant_id': participant, 'session_id': session,
                  'slice_id': slice_idx}

        return sample

    def num_elem_per_image(self):
        if self.elem_index == "mixed":
            return 1

        image = self._get_full_image()
        return image.size(self.mri_plane + 1) - \
               self.discarded_slices[0] - self.discarded_slices[1]

    def extract_slice_from_mri(self, image, index_slice):
        """
        This is a function to grab one slice in each view and create a rgb image for transferring learning: duplicate the slices into R, G, B channel
        :param image: (tensor)
        :param index_slice: (int) index of the wanted slice
        :return:
        To note, for each view:
        Axial_view = "[:, :, slice_i]"
        Coronal_view = "[:, slice_i, :]"
        Sagittal_view= "[slice_i, :, :]"
        """
        image = image.squeeze(0)
        simple_slice = image[(slice(None),) * self.mri_plane + (index_slice,)]
        triple_slice = torch.stack((simple_slice, simple_slice, simple_slice))

        return triple_slice


def return_dataset(mode, input_dir, data_df, preprocessing,
                   transformations, params, cnn_index=None):
    """
    Return appropriate Dataset according to given options.

    Args:
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
        input_dir: (str) path to a directory containing a CAPS structure.
        data_df: (DataFrame) List subjects, sessions and diagnoses.
        preprocessing: (str) type of preprocessing wanted ('t1-linear' or 't1-extensive')
        transformations: (transforms) list of transformations performed on-the-fly.
        params: (Namespace) options used by specific modes.
        cnn_index: (int) Index of the CNN in a multi-CNN paradigm (optional).

    Returns:
         (Dataset) the corresponding dataset.
    """

    if cnn_index is not None and mode in ["image", "roi", "slice"]:
        raise ValueError("Multi-CNN is not implemented for %s mode." % mode)
    if params.model == "ROI_GCN":
        use_roi = True
    else:
        use_roi = False
    if mode == "image":
        return MRIDatasetImage(
            input_dir,
            data_df,
            preprocessing,
            transformations=transformations,
            crop_padding_to_128=params.crop_padding_to_128,
            resample_size=params.resample_size,
            fake_caps_path=params.fake_caps_path,
            # only_use_fake=params.only_use_fake,
            roi=use_roi,
            roi_size=params.roi_size,
            model=params.model,
            data_preprocess=params.data_preprocess,
            data_Augmentation=params.data_Augmentation,
            method_2d=params.method_2d
        )
    if mode == "patch":
        return MRIDatasetPatch(
            input_dir,
            data_df,
            params.patch_size,
            params.stride_size,
            preprocessing=preprocessing,
            transformations=transformations,
            prepare_dl=params.prepare_dl,
            patch_index=cnn_index
        )
    elif mode == "roi":
        return MRIDatasetRoi(
            input_dir,
            data_df,
            preprocessing=preprocessing,
            transformations=transformations
        )
    elif mode == "slice":
        return MRIDatasetSlice(
            input_dir,
            data_df,
            preprocessing=preprocessing,
            transformations=transformations,
            mri_plane=params.mri_plane,
            prepare_dl=params.prepare_dl,
            discarded_slices=params.discarded_slices)
    else:
        raise ValueError("Mode %s is not implemented." % mode)


def compute_num_cnn(input_dir, tsv_path, options, data="train"):
    transformations = get_transforms(options)

    if data == "train":
        example_df, _ = load_data(tsv_path, options.diagnoses, 0, options.n_splits, options.baseline)
    elif data == "classify":
        example_df = pd.read_csv(tsv_path, sep='\t')
    else:
        example_df = load_data_test(tsv_path, options.diagnoses)

    full_dataset = return_dataset(options.mode, input_dir, example_df,
                                  options.preprocessing, transformations, options)

    return full_dataset.elem_per_image


##################################
# Transformations
##################################

class GaussianSmoothing(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        from scipy.ndimage.filters import gaussian_filter

        image = sample['image']
        np.nan_to_num(image, copy=False)
        smoothed_image = gaussian_filter(image, sigma=self.sigma)
        sample['image'] = smoothed_image

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __call__(self, image):
        np.nan_to_num(image, copy=False)
        image = image.astype(float)

        return torch.from_numpy(image[np.newaxis, :]).float()

    def __repr__(self):
        return self.__class__.__name__


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, **data_dict):
        image = data_dict['data']
        image = (image - image.min()) / (image.max() - image.min())
        data_dict['data'] = image
        return data_dict

    def __repr__(self):
        return self.__class__.__name__


class ItensityNormalizeNonzeorVolume(object):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    def __call__(self, **data_dict):
        image = data_dict['data']
        image = image.squeeze()
        image = np.array(image)
        pixels = image[image > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (image - mean) / std
        out_random = np.random.normal(0, 1, size=image.shape)

        out[image == 0] = out_random[image == 0]
        out = torch.from_numpy(out.copy())
        data_dict['data'] = out.unsqueeze(0)
        return data_dict

    def __repr__(self):
        return self.__class__.__name__


class ItensityNormalizeAllVolume(object):
    """
    normalize the itensity of an nd volume based on the mean and std of all region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    def __call__(self, **data_dict):
        image = data_dict['data']
        image = (image - image.mean()) / image.std()
        data_dict['data'] = image
        return data_dict

    def __repr__(self):
        return self.__class__.__name__


class CropPpadding128(object):
    """
    crop padding image to 128
    """

    def __call__(self, **data_dict):
        image = data_dict['data']

        if image.shape[1] == 121 and image.shape[2] == 145 and image.shape[
            3] == 121:

            image = image[:, :, 8:-9, :]  # [1, 121, 128, 121]
            image = image.unsqueeze(0)  # [1, 1, 121, 128, 121]
            pad = torch.nn.ReplicationPad3d((4, 3, 0, 0, 4, 3))
            image = pad(image)  # [1, 1, 128, 128, 128]
            image = image.squeeze(0)  # [1, 128, 128, 128]
        elif image.shape[1] == 128 and image.shape[2] == 128 and image.shape[
            3] == 128:
            pass
        else:
            assert image.shape[1] == 121 and image.shape[2] == 145 and image.shape[
                3] == 121, "image shape must be 1*121*145*122 or 1*128*128*128, but given shape:{}".format(image.shape)

        data_dict['data'] = image
        return data_dict

    def __repr__(self):
        return self.__class__.__name__


class Resize(torch.nn.Module):
    """
    Resize data to target size
    """

    def __init__(self, resample_size):
        super().__init__()
        # assert resample_size > 0, 'resample_size should be a int positive number'
        self.resample_size = resample_size

    def forward(self, **data_dict):
        image = data_dict['data']
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=self.resample_size)
        image = image.squeeze(0)
        data_dict['data'] = image
        return data_dict

    def __repr__(self):
        return self.__class__.__name__ + '(resample_size={0})'.format(self.resample_size)


class CheckDictSize(object):
    """
    check dict size to dim 5 :1,1,128,128,128
    """

    def __call__(self, **dict):
        image = dict['data']
        if len(image.shape) == 4:
            image = np.array(image.unsqueeze(0))
        elif len(image.shape) == 3:
            image = np.array(image.unsqueeze(0).unsqueeze(0))
        assert len(image.shape) == 5
        dict['data'] = image

        return dict

    def __repr__(self):
        return self.__class__.__name__


class DictToImage(object):
    """
    dict 2 data
    """

    def __call__(self, **dict):
        image = dict['data']
        if len(image.shape) == 5:
            image = image.squeeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
        return image

    def __repr__(self):
        return self.__class__.__name__


class DropInvalidRange(torch.nn.Module):
    """
    Cut off the invalid area
    """

    def __init__(self, keep_size=True):
        super().__init__()
        self.keep_size = keep_size

    def __call__(self, **data_dict):
        image = data_dict['data']
        image = image.squeeze(0)
        zero_value = image[0, 0, 0]
        z, h, w = image.shape
        non_zeros_idx = np.where(image != zero_value)
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        image = image[min_z:max_z, min_h:max_h, min_w:max_w].unsqueeze(0)
        if self.keep_size:
            image = image.unsqueeze(0)
            image = F.interpolate(image, size=[z, h, w])
            image = image.squeeze(0)
        data_dict['data'] = image
        return data_dict

    def __repr__(self):
        return self.__class__.__name__ + '(keep_size={})'.format(self.keep_size)


def get_transforms(params, is_training=True):
    if params.mode == 'image':
        trans_list = []
        trans_list.append(MinMaxNormalization())
        if params.preprocessing != 't1-linear':
            trans_list.append(CropPpadding128())
            trans_list.append(DropInvalidRange(keep_size=True))
        if params.resample_size is not None:
            trans_list.append(Resize(params.resample_size))
        if params.data_preprocess == 'MinMax':
            trans_list.append(MinMaxNormalization())
        elif params.data_preprocess == 'NonzeorZscore':
            trans_list.append(ItensityNormalizeNonzeorVolume())
        elif params.data_preprocess == 'AllzeorZscore':
            trans_list.append(ItensityNormalizeAllVolume())
        if is_training:
            if params.ContrastAugmentationTransform > 0:
                trans_list.append(CheckDictSize())  # for this code library, input data must be dim=5, 1,1,128,128,128
                trans_list.append(ContrastAugmentationTransform((0.3, 3.), preserve_range=True,
                                                                p_per_sample=params.ContrastAugmentationTransform))
            if params.BrightnessTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(
                    BrightnessTransform(mu=0, sigma=1, per_channel=False, p_per_sample=params.BrightnessTransform))
            if params.GammaTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(
                    GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=False, retain_stats=False,
                                   p_per_sample=params.GammaTransform))
            if params.BrightnessGradientAdditiveTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(BrightnessGradientAdditiveTransform(scale=(5, 5),
                                                                      p_per_sample=params.BrightnessGradientAdditiveTransform))
            if params.LocalSmoothingTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(LocalSmoothingTransform(scale=(5, 5),
                                                          p_per_sample=params.LocalSmoothingTransform))
            if params.RandomShiftTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(
                    RandomShiftTransform(shift_mu=0, shift_sigma=3, p_per_sample=params.RandomShiftTransform))
            if params.RicianNoiseTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(
                    RicianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=params.RicianNoiseTransform))
            if params.GaussianNoiseTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(
                    GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=params.RicianNoiseTransform))
            if params.GaussianBlurTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(
                    GaussianBlurTransform(blur_sigma=(1, 5), different_sigma_per_channel=False,
                                          p_per_sample=params.RicianNoiseTransform))
            if params.Rot90Transform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(Rot90Transform(num_rot=(1, 2, 3), axes=(0, 1, 2), p_per_sample=params.Rot90Transform))
            if params.MirrorTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(MirrorTransform(axes=(0, 1, 2), p_per_sample=params.MirrorTransform))
            if params.SpatialTransform > 0:
                trans_list.append(CheckDictSize())
                trans_list.append(
                    SpatialTransform(patch_size=(params.resample_size, params.resample_size, params.resample_size),
                                     p_el_per_sample=params.SpatialTransform,
                                     p_rot_per_axis=params.SpatialTransform,
                                     p_scale_per_sample=params.SpatialTransform,
                                     p_rot_per_sample=params.SpatialTransform))
        trans_list.append(DictToImage())
        transformations = Compose(trans_list)
        if params.model in ['DeepCNN', 'DeepCNN_gcn']:
            trans_list = []
            trans_list.append(MinMaxNormalization())
            trans_list.append(Resize(resample_size=[49, 39, 38]))
            trans_list.append(DictToImage())
            transformations = Compose(trans_list)
        if params.model in ['CNN2020', 'CNN2020_gcn']:
            trans_list = []
            trans_list.append(MinMaxNormalization())
            trans_list.append(Resize(resample_size=[139, 177, 144]))
            trans_list.append(ItensityNormalizeAllVolume())
            trans_list.append(DictToImage())
            transformations = Compose(trans_list)
    elif params.mode in ["patch", "roi"]:
        if params.minmaxnormalization:
            transformations = Compose([MinMaxNormalization(), DictToImage()])
        else:
            transformations = None
    elif params.mode == "slice":
        trg_size = (224, 224)
        if params.minmaxnormalization:
            transformations = transforms.Compose([MinMaxNormalization(),
                                                  transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])
        else:
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])
    else:
        raise ValueError("Transforms for mode %s are not implemented." % params.mode)
    print('transformer:{}'.format(transformations.__repr__))
    return transformations


################################
# tsv files loaders
################################

def load_data(train_val_path, diagnoses_list,
              split, n_splits=None, baseline=True, fake_caps_path=None, only_use_fake=False):
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    if n_splits is None:
        train_path = path.join(train_val_path, 'train')
        valid_path = path.join(train_val_path, 'validation')

    else:
        train_path = path.join(train_val_path, 'train_splits-' + str(n_splits),
                               'split-' + str(split))
        valid_path = path.join(train_val_path, 'validation_splits-' + str(n_splits),
                               'split-' + str(split))

    print("Train", train_path)
    print("Valid", valid_path)

    for diagnosis in diagnoses_list:
        if isinstance(baseline, str):
            if baseline in ['true', 'True']:
                train_diagnosis_path = path.join(
                    train_path, diagnosis + '_baseline.tsv')
            elif baseline in ['false', 'False']:
                train_diagnosis_path = path.join(train_path, diagnosis + '.tsv')
        else:
            if baseline:
                train_diagnosis_path = path.join(
                    train_path, diagnosis + '_baseline.tsv')
            else:
                train_diagnosis_path = path.join(train_path, diagnosis + '.tsv')
        valid_diagnosis_path = path.join(
            valid_path, diagnosis + '_baseline.tsv')

        train_diagnosis_df = pd.read_csv(train_diagnosis_path, sep='\t')
        valid_diagnosis_df = pd.read_csv(valid_diagnosis_path, sep='\t')

        train_df = pd.concat([train_df, train_diagnosis_df])
        valid_df = pd.concat([valid_df, valid_diagnosis_df])

    train_df.reset_index(inplace=True, drop=True)
    valid_df.reset_index(inplace=True, drop=True)
    if fake_caps_path is not None and not baseline:
        path_list = os.listdir(fake_caps_path)
        for t in range(len(path_list)):
            if path_list[t] != 'subjects':
                file_name = path_list[t]
        fake_tsv_path = os.path.join(fake_caps_path, file_name)

        fake_df = pd.read_csv(fake_tsv_path, sep='\t')
        train_fake_df = pd.DataFrame(columns={"participant_id": "", "session_id": "", "diagnosis": ""})
        for i in range(len(fake_df)):
            subject = fake_df.loc[i]['participant_id']
            ses_id = fake_df.loc[i]["session_id"]
            filted_df_train = train_df.loc[train_df['participant_id'] == subject].drop_duplicates().reset_index(
                drop=True)
            if filted_df_train.shape[0] != 0:
                filted_fake_df = fake_df.loc[fake_df['participant_id'] == subject].drop_duplicates().reset_index(
                    drop=True)
                diagnosis = filted_df_train.loc[0]["diagnosis"]
                filted_fake_df['diagnosis'] = diagnosis
                train_fake_df = train_fake_df.append(filted_fake_df).drop_duplicates().reset_index(drop=True)
        if only_use_fake:
            print('*** [Only] use {} fake images for train!'.format(len(train_fake_df)))
            train_df = train_fake_df
        else:
            print('use {} fake images for train!'.format(len(train_fake_df)))
            train_df = train_df.append(train_fake_df).drop_duplicates().reset_index(drop=True)
        saved_tsv_path = os.path.join(train_path, fake_caps_path.split('/')[-1])
        save_path_train = os.path.join(saved_tsv_path, 'train_real_and_fake_' + "_".join(diagnoses_list) + '.tsv')
        if not os.path.exists(saved_tsv_path):
            os.makedirs(saved_tsv_path)
        train_df.to_csv(save_path_train, sep='\t', index=False)
        print('save: {}'.format(save_path_train))
        print('train fake df:{}'.format(train_fake_df))
        print('train real and fake df:{}'.format(train_df))
        print('valid df:{}'.format(valid_df))
    else:
        print('only train real df:{}'.format(train_df))
        print('valid df:{}'.format(valid_df))

    return train_df, valid_df


def load_data_test(test_path, diagnoses_list):
    test_df = pd.DataFrame()

    for diagnosis in diagnoses_list:
        test_diagnosis_path = path.join(test_path, diagnosis + '_baseline.tsv')
        test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
        test_df = pd.concat([test_df, test_diagnosis_df])

    test_df.reset_index(inplace=True, drop=True)

    return test_df


def mix_slices(df_training, df_validation, mri_plane=0, val_size=0.15):
    """
    This is a function to gather the training and validation tsv together, then do the bad data split by slice.
    :param training_tsv:
    :param validation_tsv:
    :return:
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    df_all = pd.concat([df_training, df_validation])
    df_all = df_all.reset_index(drop=True)

    if mri_plane == 0:
        slices_per_patient = 169 - 40
        slice_index = list(np.arange(20, 169 - 20))
    elif mri_plane == 1:
        slices_per_patient = 208 - 40
        slice_index = list(np.arange(20, 208 - 20))
    else:
        slices_per_patient = 179 - 40
        slice_index = list(np.arange(20, 179 - 20))

    participant_list = list(df_all['participant_id'])
    session_list = list(df_all['session_id'])
    label_list = list(df_all['diagnosis'])

    slice_participant_list = [
        ele for ele in participant_list for _ in range(slices_per_patient)]
    slice_session_list = [
        ele for ele in session_list for _ in range(slices_per_patient)]
    slice_label_list = [
        ele for ele in label_list for _ in range(slices_per_patient)]
    slice_index_list = slice_index * len(label_list)

    df_final = pd.DataFrame(
        columns=[
            'participant_id',
            'session_id',
            'slice_id',
            'diagnosis'])
    df_final['participant_id'] = np.array(slice_participant_list)
    df_final['session_id'] = np.array(slice_session_list)
    df_final['slice_id'] = np.array(slice_index_list)
    df_final['diagnosis'] = np.array(slice_label_list)

    y = np.array(slice_label_list)
    # split the train data into training and validation set
    skf_2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=10000)
    indices = next(skf_2.split(np.zeros(len(y)), y))
    train_ind, valid_ind = indices

    df_sub_train = df_final.iloc[train_ind]
    df_sub_valid = df_final.iloc[valid_ind]

    df_sub_train.reset_index(inplace=True, drop=True)
    df_sub_valid.reset_index(inplace=True, drop=True)

    return df_sub_train, df_sub_valid
