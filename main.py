import argparse
import math

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy

from model.wrn  import WideResNet

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import sys

import logging
logging.basicConfig(filename="log/debug.log",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, ignore_index=255,
                 reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        print("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction,
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

def validate(model, val_loader, device) :
    model.eval()
    acc = 0.0
    cnt = 0
    for id, data in enumerate(val_loader) :
        #x_v, y_v = next(val_loader)
        x_v, y_v = data
        x_v = x_v.to(device)
        y_v = y_v.to(device)
        y_v_pred = model(x_v)
        prob = F.softmax(y_v_pred, dim=1)
        accu = accuracy(prob, y_v)
        acc += accu[0].item()
        cnt += 1
    return acc/cnt

def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    
    ############################################################################
    # TODO: SUPPLY your code
    ############################################################################
    criterion = CrossEntropyLoss2d(
        weight=None, ignore_index=255).cuda()
    
    param_groups = model.parameters()

    optimizer = optim.SGD(param_groups,
                            lr=args.lr,
                            weight_decay=args.wd,
                            momentum=args.momentum,
                            nesterov=False)
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_acc = 0.0
    for epoch in range(args.epoch):
        model.train()
        for it in range(args.iter_per_epoch):
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)
            ####################################################################
            # TODO: SUPPLY your code
            ####################################################################
            ###### warm-up epoch till 100 ######################
            optimizer.zero_grad()
            if epoch < args.warmup :
                y_l_pred = model(x_l) ### the logit scores
                loss_s = criterion(y_l_pred, y_l)
                loss_s.backward()
            else : ####
                ### get pseudo labels first ######
                model.eval()
                y_ul = model(x_ul)
                prob = F.softmax(y_ul, dim=1)
                val,lab = torch.max(prob, dim=1)
                mask = val > args.threshold
                x_ul_new = x_ul[mask]
                lab = lab[mask]
                model.train()
                y_l_pred = model(x_l) ### the logit scores
                loss_s = criterion(y_l_pred, y_l)
                y_ul_pred = model(x_ul_new)
                loss_us = criterion(y_ul_pred, lab)
                loss = loss_s + args.lam * loss_us
                loss.backward()

            optimizer.step()

        val_accuracy = validate(model, test_loader, device)
        loss_supmeter.add(loss_s.item())

        if val_accuracy > best_acc :
            save_dict = {
                'epoch' : epoch,
                'state_dict' : model.state_dict(),
                'optim' : optimizer.state_dict(),
                'accuracy' : val_accuracy
            }
            torch.save(save_dict, "./log/best_checkpoint.pth")

        if epoch < args.warmup :
            logging.info("sup loss, val_accuracy" + str(loss_s.item()) + " " + str(val_accuracy))
        else : 
            logging.info("sup loss, semi loss, val_accuracy" + str(loss_s.item()) + " " +  str(loss_us.item()) + " " + str(val_accuracy))
            
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Pseudo labeling \
#                                         of CIFAR10/100 with pytorch")
#     parser.add_argument("--dataset", default="cifar10", 
#                         type=str, choices=["cifar10", "cifar100"])
#     parser.add_argument("--datapath", default="./data/", 
#                         type=str, help="Path to the CIFAR-10/100 dataset")
#     parser.add_argument('--num-labeled', type=int, 
#                         default=4000, help='Total number of labeled samples')
#     parser.add_argument("--lr", default=0.03, type=float, 
#                         help="The initial learning rate") 
#     parser.add_argument("--momentum", default=0.9, type=float,
#                         help="Optimizer momentum")
#     parser.add_argument("--wd", default=0.00005, type=float,
#                         help="Weight decay")
#     parser.add_argument("--expand-labels", action="store_true", 
#                         help="expand labels to fit eval steps")
#     parser.add_argument('--train-batch', default=64, type=int,
#                         help='train batchsize')
#     parser.add_argument('--test-batch', default=64, type=int,
#                         help='train batchsize')
#     parser.add_argument('--total-iter', default=1024*512, type=int,
#                         help='total number of iterations to run')
#     parser.add_argument('--iter-per-epoch', default=1024, type=int,
#                         help="Number of iterations to run per epoch")
#     parser.add_argument('--num-workers', default=1, type=int,
#                         help="Number of workers to launch during training")
#     parser.add_argument('--threshold', type=float, default=0.95,
#                         help='Confidence Threshold for pseudo labeling')
#     parser.add_argument("--dataout", type=str, default="./path/to/output/",
#                         help="Path to save log files")
#     parser.add_argument("--model-depth", type=int, default=28,
#                         help="model depth for wide resnet") 
#     parser.add_argument("--model-width", type=int, default=2,
#                         help="model width for wide resnet")
#     parser.add_argument("--warmup", type=int, default=30,
#                         help="initial supervised warmup")
#     parser.add_argument("--savepath", type=str, default="./log/",
#                         help="initial supervised warmup")
#     parser.add_argument("--lam", type=float, default=1.0,
#                         help="weight for semi-supervised loss")
    

#     # Add more arguments if you need them
#     # Describe them in help
#     # You can (and should) change the default values of the arguments
    
#     args = parser.parse_args()

#     main(args)
