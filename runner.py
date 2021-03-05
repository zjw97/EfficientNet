import os
import copy
import time
from glob import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Runner():
    def __init__(self, arg, net, optim, torch_device, loss, logger, Visualizer, scheduler=None):
        self.arg = arg
        self.save_dir = arg.save_dir

        self.logger = logger

        self.torch_device = torch_device

        self.net = net
        self.ema = copy.deepcopy(net.module).cpu()
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.loss = loss
        self.optim = optim
        self.scheduler = scheduler

        self.start_epoch = 0
        self.best_metric = -1
        self.Visualizer = Visualizer

        # self.load()

    def save(self, epoch, filename="train"):
        """Save current epoch model

        Save Elements:
            model_type : arg.model
            start_epoch : current epoch
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score

        Parameters:
            epoch : current epoch
            filename : model save file name
        """

        torch.save({"model_type": self.arg.model,
                    "start_epoch": epoch + 1,
                    "network": self.net.module.state_dict(),
                    "ema": self.net.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("Model saved %d epoch" % (epoch))

    def load(self, filename=""):
        """ Model load. same with save"""
        if filename == "":
            # load last epoch model
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[-1])

        file_path = self.save_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File" % (self.save_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type"] != self.arg.model:
                raise ValueError("Ckpoint Model Type is %s" %
                                 (ckpoint["model_type"]))

            self.net.module.load_state_dict(ckpoint['network'])
            self.ema.load_state_dict(ckpoint['ema'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f" %
                  (ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def update_ema(self):
        with torch.no_grad():
            self.ema = copy.deepcopy(self.net)


    def train(self, train_loader, val_loader=None):
        print("\nStart Train len :", len(train_loader.dataset))
        for epoch in range(self.start_epoch, self.arg.epoch):
            self.net.train()
            train_top1 = AveragerMeter()
            for i, (input_, target_) in enumerate(train_loader):
                target_ = target_.to(self.torch_device, non_blocking=True)

                out = self.net(input_)
                loss = self.loss(out, target_)
                self.Visualizer.add_scalar("train_loss", loss, i + epoch * len(train_loader))
                self.Visualizer.add_scalar("learning_rate",self.optim.param_groups[0]['lr'], epoch)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                acc = self.accuacy(out, target_, topk=(1))
                train_top1.update(acc[0], target_.size(0))
                self.scheduler.step()

                if (i % 50) == 0:
                    self.logger.write("train[%d/%d] epoch= %d lr=%.5f  loss=%.5f  acc=%.5f{%.5f}"
                                      %(i, len(train_loader), epoch, self.optim.param_groups[0]['lr'], loss.item(), train_top1.val, train_top1.avg))

            self.logger.write("[train] precious %.3f" % (train_top1.avg))
            self.Visualizer.add_scalar("[train] acc", train_top1.avg, epoch)

            self.net.eval()
            val_top1 = AveragerMeter()
            for i, (input, target) in enumerate(val_loader):
                target = target.to(self.torch_device, non_blocking=True)
                with torch.no_grad():
                    out = self.net(input)
                    loss = self.loss(out, target)

                    acc = self.accuacy(out, target, topk=(1))
                    train_top1.update(acc[0], target.size(0))
                    self.Visualizer.add_scalar("val_loss", loss, i + epoch * len(val_loader))

                    if (i % 50) == 0:
                        self.logger.write("[val][%d/%d]  epoch= %d lr=%.5f  loss=%.5f  acc=%.5f{%.5f}"
                                          %(i, len(val_loader), epoch, self.optim.param_groups[0]['lr'], loss.item(), val_top1.val, val_top1.avg))

            self.logger.write("[val] precious %.3f"%(val_top1.avg))
            self.Visualizer.add_scalar("val_acc", val_top1.avg, i + epoch * len(val_loader))


    def accuacy(self, output, target, topk=(1, )):
        #maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(topk)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res=[]

        correct_k = correct[:topk].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def test(self, train_loader, val_loader):
        print("\n Start Test")
        self.load()
        train_acc = self._get_acc(train_loader)
        valid_acc = self._get_acc(val_loader)
        self.logger.write("[test] fname=test  train_acc= %.3f  valid_acc= %.3f "%(train_acc, valid_acc))
        return train_acc, valid_acc

class AveragerMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        self.sum = 0

    def update(self, val, n):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count
