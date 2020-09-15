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

    transformations = get_transforms(params.mode, params.minmaxnormalization)
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
                                    transformations, params, cnn_index=cnn_index)
        data_valid = return_dataset(params.mode, params.input_dir, valid_df, params.preprocessing,
                                    transformations, params, cnn_index=cnn_index)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                    batch_size=params.batch_size,
                                    shuffle=True,
                                    num_workers=params.num_workers,
                                    pin_memory=True
                                    )

        valid_loader = DataLoader(data_valid,
                                    batch_size=params.batch_size,
                                    shuffle=False,
                                    num_workers=params.num_workers,
                                    pin_memory=True
                                    )

        # Initialize the model
        print('Initialization of the model %i' % cnn_index)
        model = init_model(params.model, gpu=params.gpu, dropout=params.dropout, device_index=params.device)
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
