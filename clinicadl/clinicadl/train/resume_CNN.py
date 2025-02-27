# coding: utf8

from _typeshed import OpenBinaryMode
import argparse
from os import path
from time import time
from typing import Optional
import torch
from torch.utils.data import DataLoader

from clinicadl.test.evaluation_singleCNN import test_cnn
from clinicadl.tools.deep_learning.data import MRIDataset, MinMaxNormalization, load_data
from clinicadl.tools.deep_learning import create_model, load_model, load_optimizer, read_json
from clinicadl.tools.deep_learning.cnn_utils import train

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("model_path", type=str,
                    help="model selected")
parser.add_argument("split", type=int,
                    help="Will load the specific split wanted.")

# Computational argument
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


def main(options):
    options = read_json(options)

    if options.evaluation_steps % options.accumulation_steps != 0 and options.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (options.evaluation_steps, options.accumulation_steps))

    if options.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    total_time = time()

    # Get the data.
    training_tsv, valid_tsv = load_data(options.diagnosis_path, options.diagnoses,
                                        options.split, options.n_splits, options.baseline)

    data_train = MRIDataset(options.input_dir, training_tsv, transform=transformations,
                            preprocessing=options.preprocessing)
    data_valid = MRIDataset(options.input_dir, valid_tsv, transform=transformations,
                            preprocessing=options.preprocessing)

    # Use argument load to distinguish training and testing
    train_loader = DataLoader(data_train,
                              batch_size=options.batch_size,
                              shuffle=True,
                              num_workers=options.num_workers,
                              pin_memory=True,
                              drop_last=options.drop_last
                              )

    valid_loader = DataLoader(data_valid,
                              batch_size=options.batch_size,
                              shuffle=False,
                              num_workers=options.num_workers,
                              pin_memory=True,
                              drop_last=options.drop_last
                              )

    # Initialize the model
    print('Initialization of the model')
    if options.model == 'UNet3D':
        print('********** init UNet3D model for test! **********')
        model = create_model(options.model, gpu=options.gpu, dropout=options.dropout, device_index=options.device,
                             in_channels=options.in_channels,
                             out_channels=options.out_channels, f_maps=options.f_maps, layer_order=options.layer_order,
                             num_groups=options.num_groups, num_levels=options.num_levels)
    elif options.model == 'ResidualUNet3D':
        print('********** init ResidualUNet3D model for test! **********')
        model = create_model(options.model, gpu=options.gpu, dropout=options.dropout, device_index=options.device,
                             in_channels=options.in_channels,
                             out_channels=options.out_channels, f_maps=options.f_maps, layer_order=options.layer_order,
                             num_groups=options.num_groups, num_levels=options.num_levels)
    elif options.model == 'UNet3D_add_more_fc':
        print('********** init UNet3D_add_more_fc model for test! **********')
        model = create_model(options.model, gpu=options.gpu, dropout=options.dropout, device_index=options.device,
                             in_channels=options.in_channels,
                             out_channels=options.out_channels, f_maps=options.f_maps, layer_order=options.layer_order,
                             num_groups=options.num_groups, num_levels=options.num_levels)
    elif options.model == 'ResidualUNet3D_add_more_fc':
        print('********** init ResidualUNet3D_add_more_fc model for test! **********')
        model = create_model(options.model, gpu=options.gpu, dropout=options.dropout, device_index=options.device,
                             in_channels=options.in_channels,
                             out_channels=options.out_channels, f_maps=options.f_maps, layer_order=options.layer_order,
                             num_groups=options.num_groups, num_levels=options.num_levels)
    elif options.model == 'VoxCNN':
        print('********** init VoxCNN model for test! **********')
        model = create_model(options.model, gpu=options.gpu, device_index=options.device)
    elif options.model == 'ConvNet3D':
        print('********** init ConvNet3D model for test! **********')
        model = create_model(options.model, gpu=options.gpu, device_index=options.device)
    elif 'gcn' in options.model:
        print('********** init {}-{} model for test! **********'.format(options.model, options.gnn_type))
        model = create_model(options.model, gpu=options.gpu, device_index=options.device, gnn_type=options.gnn_type,
                             gnn_dropout=options.gnn_dropout,
                             gnn_dropout_adj=options.gnn_dropout_adj,
                             gnn_non_linear=options.gnn_non_linear,
                             gnn_undirected=options.gnn_undirected,
                             gnn_self_loop=options.gnn_self_loop,
                             gnn_threshold=options.gnn_threshold, )
    elif options.model == 'ROI_GCN':
        print('********** init ROI_GCN model for test! **********')
        model = create_model(options.model, gpu=options.gpu, device_index=options.device,
                             gnn_type=options.gnn_type,
                             gnn_dropout=options.gnn_dropout,
                             gnn_dropout_adj=options.gnn_dropout_adj,
                             gnn_non_linear=options.gnn_non_linear,
                             gnn_undirected=options.gnn_undirected,
                             gnn_self_loop=options.gnn_self_loop,
                             gnn_threshold=options.gnn_threshold,
                             nodel_vetor_layer=options.nodel_vetor_layer,
                             classify_layer=options.classify_layer,
                             num_node_features=options.num_node_features, num_class=options.num_class,
                             roi_size=options.roi_size, num_nodes=options.num_nodes,
                             gnn_pooling_layers=options.gnn_pooling_layers,
                             global_sort_pool_k=options.global_sort_pool_k,
                             layers=options.layers,
                             shortcut_type=options.shortcut_type, use_nl=options.use_nl,
                             dropout=options.dropout,
                             device=options.device)
    elif options.model == 'SwinTransformer3d':
        print('********** init SwinTransformer3d model for test! **********')
        model = create_model(options.model, gpu=options.gpu, dropout=options.dropout,
                             device_index=options.device,
                             sw_patch_size=options.sw_patch_size,
                             window_size=options.window_size,
                             mlp_ratio=options.mlp_ratio,
                             drop_rate=options.drop_rate,
                             attn_drop_rate=options.attn_drop_rate,
                             drop_path_rate=options.drop_path_rate,
                             qk_scale=options.qk_scale,
                             embed_dim=options.embed_dim,
                             depths=options.depths,
                             num_heads=options.num_heads,
                             qkv_bias=options.qkv_bias,
                             ape=options.ape,
                             patch_norm=options.patch_norm,
                             )
    elif options.model == 'vit':
        model = create_model(options.model, gpu=options.gpu, dropout=options.dropout,
                             device_index=options.device, num_class=options.num_class, args=options, )
    else:
        model = create_model(options.model, gpu=options.gpu, dropout=options.dropout, device_index=options.device)
    model_dir = path.join(options.model_path, "best_model_dir", "CNN", "fold_" + str(options.split))
    model, current_epoch = load_model(model, model_dir, options.gpu, 'checkpoint.pth.tar', device_index=options.device)

    options.beginning_epoch = current_epoch + 1

    # Define criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_path = path.join(options.model_path, 'optimizer.pth.tar')
    optimizer = load_optimizer(optimizer_path, model)

    # Define output directories
    log_dir = path.join(options.output_dir, 'log_dir', 'fold_%i' % options.split, 'CNN')
    model_dir = path.join(options.output_dir, 'best_model_dir', 'fold_%i' % options.split, 'CNN')

    print('Resuming the training task')
    train(model, train_loader, valid_loader, criterion, optimizer, True, log_dir, model_dir, options)

    options.model_path = options.output_dir
    test_cnn(train_loader, "train", options.split, criterion, options)
    test_cnn(valid_loader, "validation", options.split, criterion, options)

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
