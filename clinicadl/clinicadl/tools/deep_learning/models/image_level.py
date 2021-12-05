# coding: utf8

from .modules import PadMaxPool3d, Flatten
import torch.nn as nn
import sys
import os

sys.path.append('/root/fanchenchen/fanchenchen/MRI/code/paper')
sys.path.append('/root/fanchenchen_v2/fanchenchen/MRI/code/paper')
sys.path.append('/root/Downloads/code/MRI/paper')
sys.path.append('/root/Downloads/MRI/paper')
sys.path.append('/root/Downloads/MRI')
sys.path.append('/data/fanchenchen/Project/MRI/code/paper')
sys.path.append('/home/tian/pycharm_project/MRI_GNN/code')
sys.path.append('/home/tian/pycharm_project/MRI_GNN/code/paper')

from networks import DAM_3d, UNet3D, ResidualUNet3D, ResidualUNet3D_add_more_fc, UNet3D_add_more_fc, UNet3D_GCN
from Unet import N_Net
from ConvNet3D import ConvNet3D, ConvNet3D_gcn, ConvNet3D_v2, ConvNet3D_ori
from VoxCNN import VoxCNN, VoxCNN_gcn
from resnet50_3d import *
from ROI_GNN import ROI_GCN
from DeepCNN import DeepCNN, DeepCNN_gcn
from CNN2020 import CNN2020, CNN2020_gcn
from DenseNet_3d import *
from Dynamic2D_net import *
from swin_transformer_3d import SwinTransformer3d
from vgg import vgg_3d

__all__ = ["Conv5_FC3", "Conv5_FC3_mni", "Conv5_FC3_DAM", "Conv5_FC3_DAM_all_layer", "Conv5_FC3_DAM_last",
           "resnet50_3d", "resnet50_3d_gcn", "resnet50_3d_nl", "UNet3D", "N_Net", "ResidualUNet3D", "VoxCNN",
           "ConvNet3D", "UNet3D_add_more_fc", "ResidualUNet3D_add_more_fc", "UNet3D_GCN",
           'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200',
           'resnet101_res_fc', 'resnet101_fc', 'resnet101_mult_crop',
           'resnet10_gcn', 'resnet18_gcn', 'resnet34_gcn', 'resnet50_gcn', 'resnet101_gcn', 'resnet152_gcn',
           'resnet200_gcn', 'resnet101_fc_gcn', 'resnet101_res_fc_gcn',
           "ROI_GCN", "ConvNet3D_v2", "ConvNet3D_ori",
           "DeepCNN", "CNN2020", "ConvNet3D_gcn", "CNN2020_gcn", "VoxCNN_gcn", "DeepCNN_gcn",
           "DenseNet121_3d", "DenseNet161_3d", "DenseNet201_3d", "DenseNet121_3d_gcn", "DenseNet161_3d_gcn",
           "DenseNet201_3d_gcn", "Dynamic2D_net_Alex", "Dynamic2D_net_Res34", "Dynamic2D_net_Res18",
           "Dynamic2D_net_Vgg16", "Dynamic2D_net_Vgg11", "Dynamic2D_net_Mobile",
           "SwinTransformer3d", "vgg_3d"]

"""
All the architectures are built here
"""


# class resnet50_3d(nn.Module):
#     def __init__(self, dropout=0.5):
#         super(resnet50_3d, self).__init__()
#         self.net = i3_res50(dropout=dropout)

#     def forward(self, x):
#         x = self.net(x)
#         return x

# class resnet50_3d_nl(nn.Module):
#     def __init__(self, dropout=0.5):
#         super(resnet50_3d_nl, self).__init__()
#         self.net = i3_res50_nl(dropout=dropout)

#     def forward(self, x):
#         x = self.net(x)
#         return x


class Conv5_FC3_DAM(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """

    def __init__(self, dropout=0.5):
        super(Conv5_FC3_DAM, self).__init__()
        self.DAM = DAM_3d(1, 1)
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.DAM(x)
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv5_FC3_DAM_all_layer(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """

    def __init__(self, dropout=0.5):
        super(Conv5_FC3_DAM_all_layer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            DAM_3d(8, 8, Sagittal_plane=169, Coronal_plane=208, Axial_plane=179),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            DAM_3d(16, 16, Sagittal_plane=85, Coronal_plane=104, Axial_plane=90),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            DAM_3d(32, 32, Sagittal_plane=43, Coronal_plane=52, Axial_plane=45),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            DAM_3d(64, 64, Sagittal_plane=22, Coronal_plane=26, Axial_plane=23),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            DAM_3d(128, 128, Sagittal_plane=11, Coronal_plane=13, Axial_plane=12),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv5_FC3_DAM_last(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """

    def __init__(self, dropout=0.5):
        super(Conv5_FC3_DAM_last, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            # DAM_3d(8, 8, Sagittal_plane=169, Coronal_plane=208, Axial_plane=179),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # DAM_3d(16, 16, Sagittal_plane=85, Coronal_plane=104, Axial_plane=90),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # DAM_3d(32, 32, Sagittal_plane=43, Coronal_plane=52, Axial_plane=45),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # DAM_3d(64, 64, Sagittal_plane=22, Coronal_plane=26, Axial_plane=23),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            DAM_3d(128, 128, Sagittal_plane=11, Coronal_plane=13, Axial_plane=12),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv5_FC3(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """

    def __init__(self, dropout=0.5):
        super(Conv5_FC3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            # nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.Linear(128 * 4 * 4 * 4, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv5_FC3_mni(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Extensive preprocessing
    """

    def __init__(self, dropout=0.5):
        super(Conv5_FC3_mni, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 4 * 5 * 4, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
