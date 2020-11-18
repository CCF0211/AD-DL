from .autoencoder import AutoEncoder, initialize_other_autoencoder, transfer_learning
from .iotools import load_model, load_optimizer, save_checkpoint
from .image_level import * 
from .patch_level import Conv4_FC3
from .slice_level import resnet18
import torch
from torch import nn


def create_model(model_name, gpu=False, device_index=0, pretrain_resnet_path=None, new_layer_names=[], **kwargs):
    """
    Creates model object from the model_name.

    :param model_name: (str) the name of the model (corresponding exactly to the name of the class).
    :param gpu: (bool) if True a gpu is used.
    :return: (Module) the model object
    """

    try:
        model = eval(model_name)(**kwargs)
    except NameError:
        raise NotImplementedError(
            'The model wanted %s has not been implemented.' % model_name)

    if gpu:
        device = torch.device("cuda:{}".format(device_index))
        model.to(device)
        model = nn.DataParallel(model, device_ids=[device])
    else:
        model.cpu()
    net_dict = model.state_dict()
    
    if pretrain_resnet_path is not None:
        print ('loading pretrained model {}'.format(pretrain_resnet_path))
        pretrain = torch.load(pretrain_resnet_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        # new_parameters = [] 
        # for pname, p in model.named_parameters():
        #     for layer_name in new_layer_names:
        #         if pname.find(layer_name) >= 0:
        #             new_parameters.append(p)
        #             break

        # new_parameters_id = list(map(id, new_parameters))
        # base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        # parameters = {'base_parameters': base_parameters, 
        #               'new_parameters': new_parameters}

    return model


def create_autoencoder(model_name, gpu=False, transfer_learning_path=None, difference=0, device_index=0):
    """
    Creates an autoencoder object from the model_name.

    :param model_name: (str) the name of the model (corresponding exactly to the name of the class).
    :param gpu: (bool) if True a gpu is used.
    :param transfer_learning_path: (str) path to another pretrained autoencoder to perform transfer learning.
    :param difference: (int) difference of depth between the pretrained encoder and the new one.
    :return: (Module) the model object
    """
    from .autoencoder import AutoEncoder, initialize_other_autoencoder
    from os import path

    model = create_model(model_name, gpu, device_index=device_index)
    decoder = AutoEncoder(model)

    if transfer_learning_path is not None:
        if path.splitext(transfer_learning_path) != ".pth.tar":
            raise ValueError("The full path to the model must be given (filename included).")
        decoder = initialize_other_autoencoder(decoder, transfer_learning_path, difference)

    return decoder


def init_model(model_name, autoencoder=False, gpu=False, device_index=0, pretrain_resnet_path=None, new_layer_names=[], **kwargs):

    model = create_model(model_name, gpu=gpu, device_index=device_index, pretrain_resnet_path=pretrain_resnet_path, new_layer_names=new_layer_names, **kwargs)
    if autoencoder:
        model = AutoEncoder(model)

    return model
