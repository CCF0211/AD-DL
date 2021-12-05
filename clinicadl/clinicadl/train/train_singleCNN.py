# coding: utf8

import os
import torch
import time
from torch.utils.data import DataLoader
import wandb
import numpy as np

from ..tools.deep_learning.utils import timeSince
from ..tools.deep_learning.models import transfer_learning, init_model
from ..tools.deep_learning.data import (get_transforms,
                                        load_data,
                                        return_dataset)
from ..tools.deep_learning.cnn_utils import train
from clinicadl.test.test_singleCNN import test_cnn
from .optimizer import return_optimizer
from .criterion import return_criterion
from .lr_scheduler import return_scheduler



def train_single_cnn(params):
    """
    Trains a single CNN and writes:
        - logs obtained with Tensorboard during training,
        - best models obtained according to two metrics on the validation set (loss and balanced accuracy),
        - for patch and roi modes, the initialization state is saved as it is identical across all folds,
        - final performances at the end of the training.

    If the training crashes it is possible to relaunch the training process from the checkpoint.pth.tar and
    optimizer.pth.tar files which respectively contains the state of the model and the optimizer at the end
    of the last epoch that was completed before the crash.
    """

    train_transformations = get_transforms(params, is_training=True)
    test_transformations = get_transforms(params, is_training=False)
    train_begin_time = time.time()

    if params.split is None:
        if params.n_splits is None:
            fold_iterator = range(1)
        else:
            fold_iterator = range(params.n_splits)
    else:
        fold_iterator = [params.split]

    metric_dict_list = {}
    for fi in fold_iterator:

        training_df, valid_df = load_data(
            params.tsv_path,
            params.diagnoses,
            fi,
            n_splits=params.n_splits,
            baseline=params.baseline,
            fake_caps_path=params.fake_caps_path,
            only_use_fake=params.only_use_fake)
        data_train = return_dataset(params.mode, params.input_dir, training_df, params.preprocessing,
                                    train_transformations, params)
        data_valid = return_dataset(params.mode, params.input_dir, valid_df, params.preprocessing,
                                    test_transformations, params)
        print('use baseline:{}'.format(params.baseline))
        print(type(params.baseline))
        print('train data size:{} valid data size:{}'.format(len(data_train), len(data_valid)))
        # Use argument load to distinguish training and testing
        train_loader = DataLoader(
            data_train,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=params.drop_last
        )

        valid_loader = DataLoader(
            data_valid,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=params.drop_last
        )
        # Initialize the model
        print('Initialization of the model')
        if params.model == 'UNet3D':
            print('********** init UNet3D model! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device,
                               in_channels=params.in_channels,
                               out_channels=params.out_channels, f_maps=params.f_maps, layer_order=params.layer_order,
                               num_groups=params.num_groups, num_levels=params.num_levels,
                               pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names,
                               num_class=params.num_class)
        elif params.model == 'ResidualUNet3D':
            print('********** init ResidualUNet3D model! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device,
                               in_channels=params.in_channels,
                               out_channels=params.out_channels, f_maps=params.f_maps, layer_order=params.layer_order,
                               num_groups=params.num_groups, num_levels=params.num_levels,
                               pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names,
                               num_class=params.num_class)
        elif params.model == 'UNet3D_add_more_fc':
            print('********** init UNet3D_add_more_fc model! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device,
                               in_channels=params.in_channels,
                               out_channels=params.out_channels, f_maps=params.f_maps, layer_order=params.layer_order,
                               num_groups=params.num_groups, num_levels=params.num_levels,
                               pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names,
                               num_class=params.num_class)
        elif params.model == 'ResidualUNet3D_add_more_fc':
            print('********** init ResidualUNet3D_add_more_fc model! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device,
                               in_channels=params.in_channels,
                               out_channels=params.out_channels, f_maps=params.f_maps, layer_order=params.layer_order,
                               num_groups=params.num_groups, num_levels=params.num_levels,
                               pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names,
                               num_class=params.num_class)
        elif params.model == 'VoxCNN':
            print('********** init VoxCNN model! **********')
            model = init_model(params.model, gpu=params.gpu, device_index=params.device,
                               pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names,
                               num_class=params.num_class)
        elif params.model == 'ConvNet3D':
            print('********** init ConvNet3D model! **********')
            model = init_model(params.model, gpu=params.gpu, device_index=params.device,
                               pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names,
                               num_class=params.num_class)
        elif params.model == 'ROI_GCN':
            print('ok')
            print('********** init {}-{} model! **********'.format(params.model, params.gnn_type))
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
                               gnn_pooling_layers=params.gnn_pooling_layers,
                               global_sort_pool_k=params.global_sort_pool_k,
                               layers=params.layers,
                               shortcut_type=params.shortcut_type, use_nl=params.use_nl,
                               dropout=params.dropout,
                               device=params.device)
        elif params.model == 'SwinTransformer3d':
            print('********** init SwinTransformer3d model! **********')
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout,
                               device_index=params.device,
                               sw_patch_size=params.sw_patch_size,
                               window_size=params.window_size,
                               mlp_ratio=params.mlp_ratio,
                               drop_rate=params.drop_rate,
                               attn_drop_rate=params.attn_drop_rate,
                               drop_path_rate=params.drop_path_rate,
                               qk_scale=params.qk_scale,
                               embed_dim=params.embed_dim,
                               depths=params.depths,
                               num_heads=params.num_heads,
                               qkv_bias=params.qkv_bias,
                               ape=params.ape,
                               patch_norm=params.patch_norm,
                               num_class=params.num_class)
        elif 'gcn' in params.model:
            print('********** init {}-{} model! **********'.format(params.model, params.gnn_type))
            model = init_model(params.model, gpu=params.gpu, device_index=params.device,
                               pretrain_resnet_path=params.pretrain_resnet_path, gnn_type=params.gnn_type,
                               gnn_dropout=params.gnn_dropout,
                               gnn_dropout_adj=params.gnn_dropout_adj,
                               gnn_non_linear=params.gnn_non_linear,
                               gnn_undirected=params.gnn_undirected,
                               gnn_self_loop=params.gnn_self_loop,
                               gnn_threshold=params.gnn_threshold,
                               num_class=params.num_class)

        else:
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device,
                               pretrain_resnet_path=params.pretrain_resnet_path, new_layer_names=params.new_layer_names,
                               num_class=params.num_class)
        model = transfer_learning(model, fi, source_path=params.transfer_learning_path,
                                  gpu=params.gpu, selection=params.transfer_learning_selection,
                                  device_index=params.device)
        print(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('n_parameters:{}'.format(n_parameters))

        # Define criterion and optimizer
        criterion = return_criterion(params)
        optimizer = return_optimizer(params, model)
        lr_scheduler = return_scheduler(params, optimizer, len(train_loader))
        # criterion = torch.nn.CrossEntropyLoss()

        # wandb.watch(model, criterion, log="all", log_freq=5)

        # if params.pretrain_resnet_path is not None:
        #     new_parameters = []
        #     for pname, p in model.named_parameters():
        #         for layer_name in params.new_layer_names:
        #             if pname.find(layer_name) >= 0:
        #                 new_parameters.append(p)
        #                 break
        #
        #     new_parameters_id = list(map(id, new_parameters))
        #     base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        #     parameters = {'base_parameters': base_parameters,
        #                   'new_parameters': new_parameters}
        #     para = [
        #         {'params': parameters['base_parameters'], 'lr': params.learning_rate / 10},
        #         {'params': parameters['new_parameters'], 'lr': params.learning_rate}
        #     ]
        #     optimizer = eval("torch.optim." + params.optimizer)(para, weight_decay=params.weight_decay)
        # else:
        #     optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
        #                                                         lr=params.learning_rate,
        #                                                         weight_decay=params.weight_decay)
        setattr(params, 'beginning_epoch', 0)

        # Define output directories
        log_dir = os.path.join(
            params.output_dir, 'fold-%i' % fi, 'tensorboard_logs')
        model_dir = os.path.join(
            params.output_dir, 'fold-%i' % fi, 'models')

        print('Beginning the training task')
        train(model, train_loader, valid_loader, criterion,
              optimizer, False, log_dir, model_dir, params, fi, train_begin_time=train_begin_time,
              lr_scheduler=lr_scheduler)

        params.model_path = params.output_dir
        test_cnn(params.output_dir, train_loader, "train",
                 fi, criterion, params, gpu=params.gpu, train_begin_time=train_begin_time)
        metric_dict = test_cnn(params.output_dir, valid_loader, "validation",
                               fi, criterion, params, gpu=params.gpu, train_begin_time=train_begin_time)
        for key in metric_dict.keys():
            if key in metric_dict_list.keys():
                metric_dict_list[key].append(metric_dict[key])
            else:
                metric_dict_list[key] = [metric_dict[key]]

        torch.cuda.empty_cache()
        stop_threshold = 0.5
        if metric_dict['validation_balanced_accuracy_best_balanced_accuracy_singel_model'] < stop_threshold and fi != 4:
            print('[Early Stoped!]validation_balanced_accuracy_best_balanced_accuracy_singel_model:{}'.format(
                metric_dict['validation_balanced_accuracy_best_balanced_accuracy_singel_model']))
            wandb.log({'Early Stop fold': fi})
            break

    for keys, values in metric_dict_list.items():
        print('{}:'.format(keys))
        print(values)
    mean_matric_dict = {}
    for key in metric_dict_list.keys():
        mean_matric_dict.update({"mean_{}".format(key): np.mean(metric_dict_list[key])})
        mean_matric_dict.update({"max_{}".format(key): np.max(metric_dict_list[key])})
        mean_matric_dict.update({"std_{}".format(key): np.std(metric_dict_list[key])})
    wandb.log(mean_matric_dict)
    for keys, values in mean_matric_dict.items():
        print('{}:{}'.format(keys, values))
