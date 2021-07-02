from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.optim as optim
import torch.nn as nn

def return_criterion(args):
    if args.label_smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print('We use LabelSmoothingCrossEntropy !')
    else:
        criterion = nn.CrossEntropyLoss()
        print('We use CrossEntropy !')

    return criterion