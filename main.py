import os
import argparse

import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR

#from models.effnet import EfficientNet
from runner import Runner
from loader import get_loaders

from efficientnet_pytorch.model import EfficientNet
from logger import logger

from tensorboardX import SummaryWriter


def arg_parse():
    # projects description
    desc = "Pytorch EfficientNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--save_dir', type=str, default='output',
                        help='Directory name to save the model')

    parser.add_argument('-root', type=str, default="/home/zjw/data/data/x-ray/vinbigdata_1024/cls_data",
                        help="The Directory of data path.")
    parser.add_argument('-gpus', type=str, default="0,1",
                        help="Select GPU Numbers | 0,1,2,3 | ")
    parser.add_argument('--num_workers', type=int, default="2",
                        help="Select CPU Number workers")

    parser.add_argument('-model', type=str, default='efficientnet-b6',
                        help='The type of Efficient net.')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs')
    parser.add_argument('-batch_size', type=int, default=2, help='The size of batch')
    parser.add_argument('--test', action="store_true", help='Only Test')

    parser.add_argument('--optim', type=str, default='rmsprop', choices=["SGD", "rmsprop", "adam"])
    parser.add_argument('--lr',    type=float, default=0.256, help="Base learning rate when train batch size is 256.")

    # Adam Optimizer
    # parser.add_argument('--beta', nargs="*", type=float, default=(0.9, 0.999))

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha',      type=float, default=0.9)
    parser.add_argument('--decay',    type=float, default=1e-5)

    parser.add_argument('--scheduler', type=str, default='step',
                        help="Learning rate scheduler type")
    parser.add_argument('-step', type=int, default=30,
                        help="scheduler step size")
    return parser.parse_args()


def get_scheduler(optim, sche_type, step_size, t_max):
    if sche_type == "step":
        return StepLR(optim, step_size, gamma=0.97)
    elif sche_type == "cosine":
        return CosineAnnealingLR(optim, t_max)
    else:
        return None


if __name__ == "__main__":
    arg = arg_parse()

    time = time.time()

    arg.save_dir = "%s/%s/%d/" % (os.getcwd(), arg.save_dir, time)
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    Visualizer = SummaryWriter(arg.save_dir)

    arg.save_dir = os.path.join(arg.save_dir, "%d.log"%(int(time)))
    logger = logger(arg.save_dir)
    logger.write(str(arg) + "\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    train_loader, val_loader = get_loaders(arg.root, arg.batch_size, 528, arg.num_workers)

    net = EfficientNet.from_pretrained(arg.model, num_classes=2)
    net = nn.DataParallel(net).to(torch_device)
    loss = nn.CrossEntropyLoss()

    scaled_lr = arg.lr * arg.batch_size / 256
    optim = {
        "adam" : lambda : torch.optim.Adam(net.parameters(), lr=scaled_lr, betas=arg.beta, weight_decay=arg.decay),
        "rmsprop" : lambda : torch.optim.RMSprop(net.parameters(), lr=scaled_lr, momentum=arg.momentum, alpha=arg.alpha, weight_decay=arg.decay),
        "SGD": lambda :torch.optim.SGD(net.parameters(), lr=arg.lr, momentum=arg.momentum, weight_decay=arg.decay)
    }[arg.optim]()

    scheduler = get_scheduler(optim, arg.scheduler, int(2.4*len(train_loader)), arg.epoch * len(train_loader))

    model = Runner(arg, net, optim, torch_device, loss, logger, Visualizer, scheduler)
    if arg.test is False:
        model.train(train_loader, val_loader)
    model.test(train_loader, val_loader)
