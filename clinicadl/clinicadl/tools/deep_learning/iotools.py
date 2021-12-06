# coding: utf8


class Parameters:
    """ Class to define and initialize parameters used in training CNN networks"""

    def __init__(self, mode: str, tsv_path: str, output_dir: str, input_dir: str,
                 preprocessing: str, model: str):
        """
        Parameters:
        mode: type if input used by the network (image, patch, roi, slice)
        tsv_path: Path to the folder containing the tsv files of the
        population. To note, the column name should be participant_id,
        session_id and diagnosis.
        output_dir: Folder containing the results.
        input_dir: Path to the input folder with MRI in CAPS format.
        preprocessing: Type of preprocessing done. Choices: "t1-linear" or "t1-extensive".
        model: Neural network model.
        """
        self.mode = mode
        self.tsv_path = tsv_path
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.preprocessing = preprocessing
        self.model = model

    def write(
            self,
            transfer_learning_difference: int = 0,
            diagnoses: str = ["AD", "CN"],
            baseline: bool = False,
            minmaxnormalization: bool = False,
            n_splits: int = 1,
            split: int = 0,
            accumulation_steps: int = 1,
            epochs: int = 20,
            learning_rate: float = 1e-4,
            patience: int = 10,
            tolerance: float = 0.05,
            weight_decay: float = 1e-4,
            dropout: float = 0,
            gpu: bool = False,
            batch_size: int = 12,
            evaluation_steps: int = 0,
            num_workers: int = 1,
            transfer_learning_path: str = None,
            transfer_learning_selection: str = "best_acc",
            patch_size: int = 50,
            stride_size: int = 50,
            hippocampus_roi: bool = False,
            selection_threshold: float = 0.0,
            mri_plane: int = 0,
            discarded_slices: int = 20,
            prepare_dl: bool = False,
            visualization: bool = False,
            device: int = 0,
            patch_index: int = 0,
            in_channels: int = None,
            out_channels: int = None,
            f_maps: int = None,
            layer_order: str = None,
            num_groups: int = None,
            num_levels: int = None,
            crop_padding_to_128: bool = False,
            resample_size: int = None,
            drop_last: bool = False,
            fake_caps_path: str = None,
            only_use_fake: bool = False,
            pretrain_resnet_path: str = None,
            new_layer_names: str = [],
            gnn_type: str = '3gcn',
            nodel_vetor_layer: str = 'basic',
            classify_layer: str = 'basic',
            num_node_features: int = 512,
            num_class: int = 2,
            roi_size: int = 32,
            num_nodes: int = 116,
            layers: str = None,
            shortcut_type: str = 'B',
            use_nl: bool = False,
            gnn_dropout: float = 0,
            gnn_dropout_adj: float = 0,
            gnn_non_linear: str = 'relu',
            gnn_undirected: bool = True,
            gnn_self_loop: bool = True,
            gnn_threshold: float = 0.5,
            gnn_pooling_layers: str = 'global_mean_pool',
            global_sort_pool_k: int = 10,
            sw_patch_size: int = 3,
            window_size: int = 4,
            mlp_ratio: float = 4.,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            qk_scale: float = None,
            embed_dim: int = 96,
            depths: str = None,
            num_heads: str = None,
            qkv_bias: bool = True,
            ape: bool = False,
            patch_norm: bool = True,
            data_preprocess: str = None,
            data_Augmentation: bool = False,
            ContrastAugmentationTransform: float = 0,
            BrightnessTransform: float = 0,
            GammaTransform: float = 0,
            BrightnessGradientAdditiveTransform: float = 0,
            LocalSmoothingTransform: float = 0,
            CenterCropTransform: float = 0,
            RandomCropTransform: float = 0,
            RandomShiftTransform: float = 0,
            RicianNoiseTransform: float = 0,
            GaussianNoiseTransform: float = 0,
            GaussianBlurTransform: float = 0,
            Rot90Transform: float = 0,
            MirrorTransform: float = 0,
            SpatialTransform: float = 0,
            clip_grad: float = 5.0,
            warmup_lr: float = 5e-8,
            min_lr: float = 5e-6,
            warmup_epochs: int = 20,
            label_smoothing: float = 0.1,
            LR_scheduler: str = 'cosine',
            decay_epochs: int = 30,
            decay_rate: float = 0.1,
            optimizer: str = 'adamw',
            optimizer_eps: float = 1e-8,
            optimizer_betas: str = (0.9, 0.999),
            optimizer_momentum: float = 0.9,
            method_2d: str = None,
            VIT_model_name: str = None,
            reduce_method: str = None,

    ):
        """
        Optional parameters used for training CNN.
        transfer_learning_difference: Difference of size between the pretrained
                               autoencoder and the training.
        diagnoses: Take all the subjects possible for autoencoder training.
        baseline: Use only the baseline if True.
        minmaxnormalization: Performs MinMaxNormalization.
        n_splits: If a value is given will load data of a k-fold CV
        split: User can specify a chosen split.
        accumulation_steps: Accumulates gradients in order to increase the size
                            of the batch.
        epochs: Epochs through the data. (default=20).
        learning_rate: Learning rate of the optimization. (default=0.01).
        patience: Waiting time for early stopping.
        tolerance: Tolerance value for the early stopping.
        optimizer: Optimizer of choice for training. (default=Adam).
                   Choices=["SGD", "Adadelta", "Adam"].
        weight_decay: Weight decay of the optimizer.
        gpu: GPU usage if True.
        batch_size: Batch size for training. (default=1)
        evaluation_steps: Fix the number of batches to use before validation
        num_workers:  Define the number of batch being loaded in parallel
        transfer_learning_selection: Allow to choose from which model the weights are transferred.
                    Choices ["best_loss", "best_acc"]
        patch_size: The patch size extracted from the MRI.
        stride_size: The stride for the patch extract window from the MRI
        hippocampus_roi: If train the model using only hippocampus ROI.
        selection_threshold: Threshold on the balanced accuracies to compute
                             the subject-level performance.
        mri_plane: Which coordinate axis to take for slicing the MRI.
                   0 is for sagittal,
                   1 is for coronal and
                   2 is for axial direction
        prepare_dl: If True the outputs of preprocessing are used, else the
                    whole MRI is loaded.
                                     initialize corresponding models.
        """

        self.transfer_learning_difference = transfer_learning_difference
        self.diagnoses = diagnoses
        self.baseline = baseline
        self.minmaxnormalization = minmaxnormalization
        self.n_splits = n_splits
        self.split = split
        self.accumulation_steps = accumulation_steps
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tolerance = tolerance
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.gpu = gpu
        self.batch_size = batch_size
        self.evaluation_steps = evaluation_steps
        self.num_workers = num_workers
        self.transfer_learning_path = transfer_learning_path
        self.transfer_learning_selection = transfer_learning_selection
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.hippocampus_roi = hippocampus_roi
        self.mri_plane = mri_plane
        self.discarded_slices = discarded_slices
        self.prepare_dl = prepare_dl
        self.visualization = visualization
        self.selection_threshold = selection_threshold
        self.device = device
        self.patch_index = patch_index
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_maps = f_maps
        self.layer_order = layer_order
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.crop_padding_to_128 = crop_padding_to_128
        self.resample_size = resample_size
        self.drop_last = drop_last
        self.fake_caps_path = fake_caps_path
        self.only_use_fake = only_use_fake
        self.pretrain_resnet_path = pretrain_resnet_path
        self.new_layer_names = new_layer_names
        self.gnn_type = gnn_type
        self.nodel_vetor_layer = nodel_vetor_layer
        self.classify_layer = classify_layer
        self.num_node_features = num_node_features
        self.num_class = num_class
        self.roi_size = roi_size
        self.num_nodes = num_nodes
        self.layers = layers
        self.shortcut_type = shortcut_type
        self.use_nl = use_nl
        self.gnn_dropout = gnn_dropout
        self.gnn_dropout_adj = gnn_dropout_adj
        self.gnn_non_linear = gnn_non_linear
        self.gnn_undirected = gnn_undirected
        self.gnn_self_loop = gnn_self_loop
        self.gnn_threshold = gnn_threshold
        self.gnn_pooling_layers = gnn_pooling_layers
        self.global_sort_pool_k = global_sort_pool_k
        self.sw_patch_size = sw_patch_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.qk_scale = qk_scale
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.ape = ape
        self.patch_norm = patch_norm
        self.data_preprocess = data_preprocess
        self.data_Augmentation = data_Augmentation
        self.ContrastAugmentationTransform = ContrastAugmentationTransform
        self.BrightnessTransform = BrightnessTransform
        self.GammaTransform = GammaTransform
        self.BrightnessGradientAdditiveTransform = BrightnessGradientAdditiveTransform
        self.LocalSmoothingTransform = LocalSmoothingTransform
        self.CenterCropTransform = CenterCropTransform
        self.RandomCropTransform = RandomCropTransform
        self.RandomShiftTransform = RandomShiftTransform
        self.RicianNoiseTransform = RicianNoiseTransform
        self.GaussianNoiseTransform = GaussianNoiseTransform
        self.GaussianBlurTransform = GaussianBlurTransform
        self.Rot90Transform = Rot90Transform
        self.MirrorTransform = MirrorTransform
        self.SpatialTransform = SpatialTransform
        self.clip_grad = clip_grad
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.LR_scheduler = LR_scheduler
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.optimizer_eps = optimizer_eps
        self.optimizer_betas = optimizer_betas
        self.optimizer_momentum = optimizer_momentum
        self.method_2d = method_2d
        self.VIT_model_name = VIT_model_name
        self.reduce_method = reduce_method


def check_and_clean(d):
    import shutil
    import os

    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def commandline_to_json(commandline, is_args=False):
    """
    This is a function to write the python argparse object into a json file.
    This helps for DL when searching for hyperparameters

    :param commandline: a tuple contain the output of
                        `parser.parse_known_args()`

    :return:
    """
    import json
    import os
    if is_args:
        commandline_arg_dic = vars(commandline)
    else:
        commandline_arg_dic = vars(commandline[0])
        commandline_arg_dic['unknown_arg'] = commandline[1]

    output_dir = commandline_arg_dic['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # remove these entries from the commandline log file
    # if 'func' in commandline_arg_dic:
    #     del commandline_arg_dic['func']

    # if 'caps_dir' in commandline_arg_dic:
    #     del commandline_arg_dic['caps_dir']

    # if 'tsv_path' in commandline_arg_dic:
    #     del commandline_arg_dic['tsv_path']

    # if 'output_dir' in commandline_arg_dic:
    #     del commandline_arg_dic['output_dir']

    # save to json file
    json = json.dumps(commandline_arg_dic, skipkeys=True, indent=4)
    print("Path of json file:", os.path.join(output_dir, "commandline.json"))
    f = open(os.path.join(output_dir, "commandline.json"), "w")
    f.write(json)
    f.close()


def read_json(options, json_path=None, test=False, read_all_para=False):
    """
    Read a json file to update python argparse Namespace.

    Args:
        options: (argparse.Namespace) options of the model.
        json_path: (str) If given path to the json file, else found with options.model_path.
        test: (bool) If given the reader will ignore some options specific to data.
    Returns:
        options (args.Namespace) options of the model updated
    """
    import json
    from os import path
    from ...cli import set_default_dropout

    evaluation_parameters = ["diagnosis_path", "input_dir", "diagnoses"]
    prep_compatibility_dict = {"mni": "t1-extensive", "linear": "t1-linear"}
    if json_path is None:
        json_path = path.join(options.model_path, 'commandline.json')

    with open(json_path, "r") as f:
        json_data = json.load(f)
    if read_all_para:
        for key, item in json_data.items():
            # We do not change json to save the true json path
            if key in ['json']:
                pass
            else:
                setattr(options, key, item)

    else:
        for key, item in json_data.items():
            # We do not change computational options
            if key in ['gpu', 'device', 'num_workers', 'num_threads']:
                pass
            # If used for evaluation, some parameters were already given
            if test and key in evaluation_parameters:
                pass
            else:
                setattr(options, key, item)

    # Retro-compatibility with runs of previous versions
    if not hasattr(options, "model"):
        options.model = options.network
        del options.network

    if not hasattr(options, 'dropout'):
        options.dropout = None
    set_default_dropout(options)

    if not hasattr(options, 'discarded_sliced'):
        options.discarded_slices = 20

    if options.preprocessing in prep_compatibility_dict.keys():
        options.preprocessing = prep_compatibility_dict[options.preprocessing]

    if hasattr(options, 'mri_plane'):
        options.slice_direction = options.mri_plane
        del options.mri_plane

    if hasattr(options, "hippocampus_roi"):
        if options.hippocampus_roi:
            options.mode = "roi"
            del options.hippocampus_roi

    if hasattr(options, "pretrained_path"):
        options.transfer_learning_path = options.pretrained_path
        del options.pretrained_path

    if hasattr(options, "pretrained_difference"):
        options.transfer_learning_difference = options.pretrained_difference
        del options.pretrained_difference

    if hasattr(options, 'slice_direction'):
        options.mri_plane = options.slice_direction

    if hasattr(options, 'patch_stride'):
        options.stride_size = options.patch_stride

    if hasattr(options, 'use_gpu'):
        options.use_cpu = not options.use_gpu

    if hasattr(options, 'use_extracted_patches'):
        options.prepare_dl = not options.use_extracted_patches

    if options.mode == "subject":
        options.mode = "image"
    if options.mode == "slice" and not hasattr(options, "mode_task"):
        options.mode_task = "cnn"
    if options.mode == "patch" and hasattr(options, "network_type"):
        if options.network_type == "multi":
            options.mode_task = "multicnn"
        del options.network_type

    if not hasattr(options, "mode_task"):
        if hasattr(options, "train_autoencoder"):
            options.mode_task = "autoencoder"
        else:
            options.mode_task = "cnn"

    if hasattr(options, "use_cpu"):
        options.gpu = not options.use_cpu

    if hasattr(options, "unnormalize"):
        options.minmaxnormalization = not options.unnormalize

    if hasattr(options, "selection"):
        options.transfer_learning_selection = options.selection

    if hasattr(options, "use_extracted_slices"):
        options.prepare_dl = options.use_extracted_slices
    if hasattr(options, "use_extracted_patches"):
        options.prepare_dl = options.use_extracted_patches
    if hasattr(options, "use_extracted_roi"):
        options.prepare_dl = options.use_extracted_roi

    return options


def memReport():
    import gc
    import torch

    cnt_tensor = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size(), obj.is_cuda)
            cnt_tensor += 1
    print('Count: ', cnt_tensor)


def cpuStats():
    import sys
    import psutil
    import os

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


def print_commandline(commandline):
    """
    print commandline
    """
    import json
    import os

    commandline_arg_dic = vars(commandline)

    # save to json file
    # json = json.dumps(commandline_arg_dic, skipkeys=True, indent=4)
    print(json.dumps(commandline_arg_dic, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
