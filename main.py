import util.data_utils as pu
import util.utils as ut
from torch.utils.data import DataLoader
import argparse
import torch
from torchvision import models, transforms
from model.CNNs import FineTuneModel
from model import model_utils as mu

from config import *

def main(args):
    
    log = ut.Logger()
    log.open(OUTPUT_FILE, mode='w')
    
    input_trans = transforms.Compose([
            transforms.Lambda(lambda x: ut.resize_cv2(x, args.im_h, args.im_w)),
        ])
    dset_train = pu.DataLoader(TRAIN_CSV_FILE, ROOT_DIR, input_trans)
    
    labels = dset_train.classes  # all lables
    num_classes = dset_train.num_classes # number of classes
    encoder = dset_train.encoder # encoder
    
    dset_valid = pu.DataLoader(VALID_CSV_FILE, ROOT_DIR, input_trans, labels)
        
    #----------------------------------Configure------------------------------#
    # model arquitechture
    if args.pretrained:
        log.write("=> using pre-trained model '{}'\n".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        log.write("=> creating model '{}'\n".format(args.arch))
        model = models.__dict__[args.arch]()
        
    model = FineTuneModel(model, args.arch, num_classes)
    # optimizer 
    optimizer = torch.optim.SGD(model.parameters(), 
                                 args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    # criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    if GPU_AVAIL:
        model = model.cuda()
        criterion = criterion.cuda()
        log.write("Using GPU...\n")
    
    #-----------------------------Data augmentation---------------------------#
    train_loader = DataLoader(dset_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=GPU_AVAIL)    
    valid_loader = DataLoader(dset_valid,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=GPU_AVAIL)
    
    #-----------------------------Training model -----------------------------#
    model = mu.train(train_loader, valid_loader, model, criterion, optimizer, args,log)
    
    #-------------------------------- Testing --------------------------------#
    encoder = dset_train.encoder # encoder
    return 0


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                if name.islower() and not name.startswith("__")
                   and callable(models.__dict__[name]))
    
    prs = argparse.ArgumentParser(description='Fish challenge')
    prs.add_argument('-message', default=' ', type=str, help='Message to describe experiment in spreadsheet')
    prs.add_argument('-im_h', default=224, type=int, help='image height (default: 256)')
    prs.add_argument('-im_w', default=224, type=int, help='image width (default: 256)')
    prs.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', 
                        choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    prs.add_argument('-epochs', default=100, type=int, help='Number of total epochs to run')
    prs.add_argument('-lr_patience', default=3, type=int, help='Number of patience to update lr')
    prs.add_argument('-early_stop', default=5, type=int, help='Early stopping')
    prs.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='Number of data loading workers')
    prs.add_argument('-lr', '--lr', default=0.01, type=float, metavar='LR', help='Initial learning rate')
    prs.add_argument('-b', '--batch_size', default=2, type=int, metavar='N', help='Mini-batch size (default: 16)')
    prs.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    prs.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    prs.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    
    args = prs.parse_args()
    
    main(args)
    print('running correctly')

#ps.imshow()