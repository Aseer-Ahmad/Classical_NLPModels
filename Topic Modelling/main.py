import argparse
import math

from dataloader import get_cifar10, get_cifar100
from vat        import VATLoss
from utils      import accuracy
from model.wrn  import WideResNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data   import DataLoader
import torch.nn.functional as F
import logging
import os
from datetime import datetime


filename = "log/debug.log"
os.makedirs(os.path.dirname(filename), exist_ok = True)

logging.basicConfig(filename="log/debug.log",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
                            
		
def validate(model, val_loader, device, criterion) :
    model.eval()
    acc = 0.0
    loss = 0.0
    cnt = 0
    for id, data in enumerate(val_loader) :
        x_v, y_v = data
        x_v = x_v.to(device)
        y_v = y_v.to(device)
        y_v_pred = model(x_v)
        prob = F.softmax(y_v_pred, dim=1)
        accu = accuracy(prob, y_v)
        acc += accu[0].item()
        loss_v = criterion(y_v_pred, y_v)
        loss += loss_v.item()
        cnt += 1
    return acc/cnt, loss/cnt
    
def load_ckp(checkpoint_fpath, model, optimizer):
    epoch = 0
    if os.path.exists(checkpoint_fpath):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim'])
        epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

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
    best_acc = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    param_groups = model.parameters()

    optimizer = optim.SGD(param_groups,
                            lr=args.lr,
                            weight_decay=args.wd)
                            # momentum=args.momentum,
                            # nesterov=False)
    
    model, optimizer , r_epoch = load_ckp("./log/best_checkpoint.pth", model, optimizer)
    model.train()

    for epoch in range(r_epoch, args.epoch):
        for i in range(args.iter_per_epoch):
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
            vatloss = VATLoss(args)
            
            optimizer.zero_grad()
            y_l_out = model(x_l)
            y_l_pred = F.softmax(y_l_out, dim=1)
            accu = accuracy(y_l_pred, y_l)
            loss_s = criterion(y_l_out, y_l)
            
            if epoch < args.warmup :
                loss = loss_s
            else :
                loss = loss_s + args.alpha * vatloss(model, x_ul)

            loss.backward()
            optimizer.step()
            
        val_accuracy, val_loss = validate(model, test_loader, device, criterion)
        
        if val_accuracy > best_acc :
            save_dict = {
            'epoch' : epoch,
            'state_dict' : model.state_dict(),
            'optim' : optimizer.state_dict(),
            'accuracy' : val_accuracy
            }
            torch.save(save_dict, "./log/best_checkpoint.pth")
            best_acc = val_accuracy
            print(f"saving on epoch :{epoch} with val accuracy {best_acc}")
        
        logging.info(f"epoch : {epoch} warmup : {str(epoch < args.warmup)} train loss : {loss.item()} train acc {accu}  val loss {val_loss} val_accuracy {val_accuracy} ")
        print(f"{ datetime.now().time() } epoch : {epoch} warmup : {str(epoch < args.warmup)} train loss : {loss.item()} train acc {accu}  val loss {val_loss} val_accuracy {val_accuracy} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual Adverserial Training \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.009, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1024*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")                        
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=4,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=10.0, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=1.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument("--vat-iter", default=1, type=int, 
                        help="VAT iteration parameter") 
    parser.add_argument("--warmup", type = int, default = 30, 
                        help = "initial supervised warmup")
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)
    

