# coding: utf8

import os
import torch
import time
from torch.utils.data import DataLoader

from ..tools.deep_learning.utils import timeSince
from ..tools.deep_learning.models import transfer_learning, init_model
from ..tools.deep_learning.data import (get_transforms,
                                        load_data,
                                        return_dataset,
                                        compute_num_cnn)
from ..tools.deep_learning.cnn_utils import train, soft_voting_to_tsvs
from clinicadl.test.test_multiCNN import test_cnn
import wandb
import numpy as np

def train_select_patch(params):
    """
    Trains one CNN for select patch.

    If the training crashes it is possible to relaunch the training process from the checkpoint.pth.tar and
    optimizer.pth.tar files which respectively contains the state of the model and the optimizer at the end
    of the last epoch that was completed before the crash.
    """

    train_transformations = get_transforms(params, is_training=True)
    test_transformations = get_transforms(params, is_training=False)
    train_begin_time = time.time()

    num_cnn = compute_num_cnn(params.input_dir, params.tsv_path, params, data="train")

    if params.split is None:
        if params.n_splits is None:
            fold_iterator = range(1)
        else:
            fold_iterator = range(params.n_splits)
    else:
        fold_iterator = [params.split]
    matric_dict_list={}
    # Loop on folds
    for fi in fold_iterator:
        print("Fold %i" % fi)
        print("patch_index %i" % fi)

        cnn_index = params.patch_index

        training_df, valid_df = load_data(
            params.tsv_path,
            params.diagnoses,
            fi,
            n_splits=params.n_splits,
            baseline=params.baseline)

        data_train = return_dataset(params.mode, params.input_dir, training_df, params.preprocessing,
                                    train_transformations, params, cnn_index=cnn_index)
        data_valid = return_dataset(params.mode, params.input_dir, valid_df, params.preprocessing,
                                    test_transformations, params, cnn_index=cnn_index)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                    batch_size=params.batch_size,
                                    shuffle=True,
                                    num_workers=params.num_workers,
                                    pin_memory=True,
                                    drop_last=params.drop_last
                                    )

        valid_loader = DataLoader(data_valid,
                                    batch_size=params.batch_size,
                                    shuffle=False,
                                    num_workers=params.num_workers,
                                    pin_memory=True,
                                    drop_last=params.drop_last
                                    )

        # Initialize the model
        print('Initialization of the model %i' % cnn_index)
        if params.model == 'UNet3D':
            print('********** init UNet3D model! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device, in_channels=params.in_channels,
                 out_channels=params.out_channels, f_maps=params.f_maps, layer_order=params.layer_order, num_groups=params.num_groups, num_levels=params.num_levels, pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names)
        elif params.model == 'ResidualUNet3D':
            print('********** init ResidualUNet3D model! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device, in_channels=params.in_channels,
                 out_channels=params.out_channels, f_maps=params.f_maps, layer_order=params.layer_order, num_groups=params.num_groups, num_levels=params.num_levels, pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names)
        elif params.model == 'UNet3D_add_more_fc':
            print('********** init UNet3D_add_more_fc model! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device, in_channels=params.in_channels,
                 out_channels=params.out_channels, f_maps=params.f_maps, layer_order=params.layer_order, num_groups=params.num_groups, num_levels=params.num_levels, pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names)
        elif params.model == 'ResidualUNet3D_add_more_fc':
            print('********** init ResidualUNet3D_add_more_fc model! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device, in_channels=params.in_channels,
                 out_channels=params.out_channels, f_maps=params.f_maps, layer_order=params.layer_order, num_groups=params.num_groups, num_levels=params.num_levels, pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names)       
        elif params.model == 'VoxCNN':
            print('********** init VoxCNN model! **********')
            model = init_model(params.model, gpu=params.gpu, device_index=params.device, pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names)
        elif params.model == 'ConvNet3D':
            print('********** init ConvNet3D model! **********')
            model = init_model(params.model, gpu=params.gpu, device_index=params.device, pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names)
        elif 'gcn' in params.model:
            print('********** init {}-{} model! **********'.format(params.model, params.gnn_type))
            model = init_model(params.model, gpu=params.gpu, device_index=params.device,
                                pretrain_resnet_path=params.pretrain_resnet_path, gnn_type=params.gnn_type,
                                gnn_dropout=params.gnn_dropout, 
                                gnn_dropout_adj=params.gnn_dropout_adj,
                                gnn_non_linear=params.gnn_non_linear, 
                                gnn_undirected=params.gnn_undirected, 
                                gnn_self_loop=params.gnn_self_loop,
                                gnn_threshold=params.gnn_threshold,)
        elif params.model == 'ROI_GCN':
            print('********** init ROI_GCN model for test! **********')
            model = init_model(params.model, gpu=params.gpu, device_index=params.device,
                                    gnn_type=params.gnn_type,
                                    gnn_dropout=params.gnn_dropout, 
                                    gnn_dropout_adj=params.gnn_dropout_adj,
                                    gnn_non_linear=params.gnn_non_linear, 
                                    gnn_undirected=params.gnn_undirected, 
                                    gnn_self_loop=params.gnn_self_loop,
                                    gnn_threshold=params.gnn_threshold,
                                    nodel_vetor_layer=params.nodel_vetor_layer,
                                    classify_layer=params.classify_layer,
                                    num_node_features=params.num_node_features, num_class=params.num_class,
                                    roi_size=params.roi_size, num_nodes=params.num_nodes,
                                    gnn_pooling_layers=params.gnn_pooling_layers, global_sort_pool_k=params.global_sort_pool_k,
                                    layers=params.layers,
                                    shortcut_type=params.shortcut_type, use_nl=params.use_nl,
                                    dropout=params.dropout,
                                    device=params.device)
        elif params.model == 'SwinTransformer3d':
            print('********** ! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout,
                            device_index=params.device, 
                            sw_patch_size=params.sw_patch_size, 
                            window_size = params.window_size,
                            mlp_ratio = params.mlp_ratio,
                            drop_rate = params.drop_rate,
                            attn_drop_rate = params.attn_drop_rate,
                            drop_path_rate = params.drop_path_rate,
                            qk_scale = params.qk_scale,
                            embed_dim = params.embed_dim,
                            depths = params.depths,
                            num_heads = params.num_heads,
                            qkv_bias = params.qkv_bias,
                            ape = params.ape,
                            patch_norm = params.patch_norm,
                            )
        else:
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device, pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names)
        model = transfer_learning(model, fi, source_path=params.transfer_learning_path,
                                    gpu=params.gpu, selection=params.transfer_learning_selection, device_index=params.device)

        # Define criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                            lr=params.learning_rate,
                                                            weight_decay=params.weight_decay)
        setattr(params, 'beginning_epoch', 0)

        # Define output directories
        log_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'tensorboard_logs', "cnn-%i" % cnn_index,)
        model_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'models', "cnn-%i" % cnn_index)

        print('Beginning the training task')
        train(model, train_loader, valid_loader, criterion, optimizer, False, log_dir, model_dir, params, fi, cnn_index, num_cnn, train_begin_time=train_begin_time)

        test_cnn(params.output_dir, train_loader, "train", fi, criterion, cnn_index, params, gpu=params.gpu, train_begin_time=train_begin_time)
        metric_dict = test_cnn(params.output_dir, valid_loader, "validation", fi, criterion, cnn_index, params, gpu=params.gpu, train_begin_time=train_begin_time)

        answer_report = metric_dict['validation_balanced_accuracy_best_balanced_accuracy_singel_model']

    #     for key in metric_dict.keys():
    #         if key in matric_dict_list.keys():
    #             matric_dict_list[key].append(metric_dict[key])
    #         else:
    #             matric_dict_list[key] = [metric_dict[key]]

    # for keys,values in matric_dict_list.items():
    #     print('{}:'.format(keys))
    #     print(values)
    # mean_matric_dict = {}
    # for key in matric_dict_list.keys():
    #     mean_matric_dict.update({"mean_{}".format(key): np.mean(matric_dict_list[key])})
    # wandb.log(mean_matric_dict)
    # for keys,values in mean_matric_dict.items():
    #     print('{}:{}'.format(keys,values))
    return answer_report
