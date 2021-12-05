# coding: utf8

from __future__ import print_function
import argparse
import os
from os import path
import torch.nn as nn
from torch.utils.data import DataLoader

from clinicadl.tools.deep_learning.utils import timeSince
from clinicadl.tools.deep_learning.data import return_dataset, get_transforms, load_data_test
from clinicadl.tools.deep_learning import read_json, create_model, load_model
from clinicadl.tools.deep_learning.cnn_utils import test, mode_level_to_tsvs, soft_voting_to_tsvs
import wandb


def test_cnn(output_dir, data_loader, subset_name, split, criterion, model_options, gpu=False, train_begin_time=None):
    metric_dict = {}
    for selection in ["best_balanced_accuracy", "best_loss"]:
        # load the best trained model during the training
        if model_options.model == 'UNet3D':
            print('********** init UNet3D model for test! **********')
            model = create_model(model_options.model, gpu=model_options.gpu, dropout=model_options.dropout,
                                 device_index=model_options.device, in_channels=model_options.in_channels,
                                 out_channels=model_options.out_channels, f_maps=model_options.f_maps,
                                 layer_order=model_options.layer_order, num_groups=model_options.num_groups,
                                 num_levels=model_options.num_levels, num_class=model_options.num_class)
        elif model_options.model == 'ResidualUNet3D':
            print('********** init ResidualUNet3D model for test! **********')
            model = create_model(model_options.model, gpu=model_options.gpu, dropout=model_options.dropout,
                                 device_index=model_options.device, in_channels=model_options.in_channels,
                                 out_channels=model_options.out_channels, f_maps=model_options.f_maps,
                                 layer_order=model_options.layer_order, num_groups=model_options.num_groups,
                                 num_levels=model_options.num_levels, num_class=model_options.num_class)
        elif model_options.model == 'UNet3D_add_more_fc':
            print('********** init UNet3D_add_more_fc model for test! **********')
            model = create_model(model_options.model, gpu=model_options.gpu, dropout=model_options.dropout,
                                 device_index=model_options.device, in_channels=model_options.in_channels,
                                 out_channels=model_options.out_channels, f_maps=model_options.f_maps,
                                 layer_order=model_options.layer_order, num_groups=model_options.num_groups,
                                 num_levels=model_options.num_levels, num_class=model_options.num_class)
        elif model_options.model == 'ResidualUNet3D_add_more_fc':
            print('********** init ResidualUNet3D_add_more_fc model for test! **********')
            model = create_model(model_options.model, gpu=model_options.gpu, dropout=model_options.dropout,
                                 device_index=model_options.device, in_channels=model_options.in_channels,
                                 out_channels=model_options.out_channels, f_maps=model_options.f_maps,
                                 layer_order=model_options.layer_order, num_groups=model_options.num_groups,
                                 num_levels=model_options.num_levels, num_class=model_options.num_class)
        elif model_options.model == 'VoxCNN':
            print('********** init VoxCNN model for test! **********')
            model = create_model(model_options.model, gpu=model_options.gpu, device_index=model_options.device,
                                 num_class=model_options.num_class)
        elif model_options.model == 'ConvNet3D':
            print('********** init ConvNet3D model for test! **********')
            model = create_model(model_options.model, gpu=model_options.gpu, device_index=model_options.device,
                                 num_class=model_options.num_class)
        elif 'gcn' in model_options.model:
            print(
                '********** init {}-{} model for test! **********'.format(model_options.model, model_options.gnn_type))
            model = create_model(model_options.model, gpu=model_options.gpu, device_index=model_options.device,
                                 gnn_type=model_options.gnn_type,
                                 gnn_dropout=model_options.gnn_dropout,
                                 gnn_dropout_adj=model_options.gnn_dropout_adj,
                                 gnn_non_linear=model_options.gnn_non_linear,
                                 gnn_undirected=model_options.gnn_undirected,
                                 gnn_self_loop=model_options.gnn_self_loop,
                                 gnn_threshold=model_options.gnn_threshold,
                                 num_class=model_options.num_class)
        elif model_options.model == 'ROI_GCN':
            print('********** init ROI_GCN model for test! **********')
            model = create_model(model_options.model, gpu=model_options.gpu, device_index=model_options.device,
                                 gnn_type=model_options.gnn_type,
                                 gnn_dropout=model_options.gnn_dropout,
                                 gnn_dropout_adj=model_options.gnn_dropout_adj,
                                 gnn_non_linear=model_options.gnn_non_linear,
                                 gnn_undirected=model_options.gnn_undirected,
                                 gnn_self_loop=model_options.gnn_self_loop,
                                 gnn_threshold=model_options.gnn_threshold,
                                 nodel_vetor_layer=model_options.nodel_vetor_layer,
                                 classify_layer=model_options.classify_layer,
                                 num_node_features=model_options.num_node_features, num_class=model_options.num_class,
                                 roi_size=model_options.roi_size, num_nodes=model_options.num_nodes,
                                 gnn_pooling_layers=model_options.gnn_pooling_layers,
                                 global_sort_pool_k=model_options.global_sort_pool_k,
                                 layers=model_options.layers,
                                 shortcut_type=model_options.shortcut_type, use_nl=model_options.use_nl,
                                 dropout=model_options.dropout,
                                 device=model_options.device)
        elif model_options.model == 'SwinTransformer3d':
            print('********** init SwinTransformer3d model for test! **********')
            model = create_model(model_options.model, gpu=model_options.gpu, dropout=model_options.dropout,
                                 device_index=model_options.device,
                                 sw_patch_size=model_options.sw_patch_size,
                                 window_size=model_options.window_size,
                                 mlp_ratio=model_options.mlp_ratio,
                                 drop_rate=model_options.drop_rate,
                                 attn_drop_rate=model_options.attn_drop_rate,
                                 drop_path_rate=model_options.drop_path_rate,
                                 qk_scale=model_options.qk_scale,
                                 embed_dim=model_options.embed_dim,
                                 depths=model_options.depths,
                                 num_heads=model_options.num_heads,
                                 qkv_bias=model_options.qkv_bias,
                                 ape=model_options.ape,
                                 patch_norm=model_options.patch_norm,
                                 num_class=model_options.num_class)
        else:
            print('********** init model for test! **********')
            model = create_model(model_options.model, gpu=model_options.gpu, dropout=model_options.dropout,
                                 device_index=model_options.device, num_class=model_options.num_class)
        model, best_epoch = load_model(model, os.path.join(output_dir, 'fold-%i' % split, 'models', selection),
                                       gpu=gpu, filename='model_best.pth.tar', device_index=model_options.device)
        results_df, metrics = test(model, data_loader, gpu, criterion, model_options.mode,
                                   device_index=model_options.device, train_begin_time=train_begin_time,
                                   model_options=model_options, fi=split)
        print("[%s]: %s level balanced accuracy is %f" % (
            timeSince(train_begin_time), model_options.mode, metrics['balanced_accuracy']))
        print('[{}]: {}_{}_result_df:'.format(timeSince(train_begin_time), subset_name, selection))
        print(results_df)
        print('[{}]: {}_{}_metrics:\n{}'.format(timeSince(train_begin_time), subset_name, selection, metrics))
        wandb.log({'{}_accuracy_{}_singel_model'.format(subset_name, selection): metrics['accuracy'],
                   '{}_balanced_accuracy_{}_singel_model'.format(subset_name, selection): metrics['balanced_accuracy'],
                   '{}_sensitivity_{}_singel_model'.format(subset_name, selection): metrics['sensitivity'],
                   '{}_specificity_{}_singel_model'.format(subset_name, selection): metrics['specificity'],
                   '{}_precision_{}_singel_model'.format(subset_name, selection): metrics['precision'],
                   '{}_recall_{}_singel_model'.format(subset_name, selection): metrics['recall'],
                   '{}_f1_{}_singel_model'.format(subset_name, selection): metrics['f1'],
                   # '{}_roc_auc_{}_singel_model'.format(subset_name, selection): metrics['roc_auc'],
                   '{}_ppv_{}_singel_model'.format(subset_name, selection): metrics['ppv'],
                   '{}_npv_{}_singel_model'.format(subset_name, selection): metrics['npv'],
                   '{}_total_loss_{}_singel_model'.format(subset_name, selection): metrics['total_loss'],
                   })

        mode_level_to_tsvs(output_dir, results_df, metrics, split, selection, model_options.mode, dataset=subset_name)

        # Soft voting
        if model_options.mode in ["patch", "roi", "slice"]:
            soft_voting_to_tsvs(output_dir, split, selection=selection, mode=model_options.mode, dataset=subset_name,
                                selection_threshold=model_options.selection_threshold)
        # return metric dict
        metric_temp_dict = {'{}_accuracy_{}_singel_model'.format(subset_name, selection): metrics['accuracy'],
                            '{}_balanced_accuracy_{}_singel_model'.format(subset_name, selection): metrics[
                                'balanced_accuracy'],
                            '{}_sensitivity_{}_singel_model'.format(subset_name, selection): metrics['sensitivity'],
                            '{}_specificity_{}_singel_model'.format(subset_name, selection): metrics['specificity'],
                            '{}_precision_{}_singel_model'.format(subset_name, selection): metrics['precision'],
                            '{}_recall_{}_singel_model'.format(subset_name, selection): metrics['recall'],
                            '{}_f1_{}_singel_model'.format(subset_name, selection): metrics['f1'],
                            # '{}_roc_auc_{}_singel_model'.format(subset_name, selection): metrics['roc_auc'],
                            '{}_ppv_{}_singel_model'.format(subset_name, selection): metrics['ppv'],
                            '{}_npv_{}_singel_model'.format(subset_name, selection): metrics['npv'],
                            '{}_total_loss_{}_singel_model'.format(subset_name, selection): metrics['total_loss'],
                            }
        metric_dict.update(metric_temp_dict)
    return metric_dict


parser = argparse.ArgumentParser(description="Argparser for evaluation of classifiers")

# Mandatory arguments
parser.add_argument("model_path", type=str,
                    help="Path to the trained model folder.")
parser.add_argument("caps_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("tsv_path", type=str,
                    help="Path to the folder containing the tsv files of the population.")
parser.add_argument("dataset", type=str,
                    help="Name of the dataset on which the classification is performed.")

# Data Management
parser.add_argument("--diagnoses", default=None, type=str, nargs='+',
                    help='Default will load the same diagnoses used in training.')

# Computational resources
parser.add_argument("--batch_size", default=16, type=int,
                    help='Size of the batch loaded by the data loader.')
parser.add_argument("--num_workers", '-w', default=8, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--gpu", action="store_true", default=False,
                    help="if True computes the visualization on GPU")

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])

    # Read json
    model_options = argparse.Namespace()
    json_path = path.join(options.model_path, "commandline_cnn.json")
    model_options = read_json(model_options, json_path=json_path)

    # Load test data
    if options.diagnoses is None:
        options.diagnoses = model_options.diagnoses

    test_df = load_data_test(options.tsv_path, options.diagnoses)
    test_transformations = get_transforms(options, is_training=False)
    criterion = nn.CrossEntropyLoss()

    # Loop on all folds trained
    best_model_dir = os.path.join(options.model_path, 'best_model_dir')
    folds_dir = os.listdir(best_model_dir)

    for fold_dir in folds_dir:
        split = int(fold_dir[-1])
        print("Fold %i" % split)

        dataset = return_dataset(model_options.mode, options.input_dir, test_df, options.preprocessing,
                                 test_transformations, model_options)
        test_loader = DataLoader(
            dataset,
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            pin_memory=True,
            drop_last=options.drop_last
        )

        test_cnn(options.model_path, test_loader, options.dataset, split, criterion,
                 model_options, options.gpu)
