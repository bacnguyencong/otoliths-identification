import util.data_utils as pu
import util.utils as ut
from torch.utils.data import DataLoader
import argparse
import torch
import PIL
from torchvision import models, transforms
from model.CNNs import FineTuneModel
from model import model_utils as mu
from torchvision.datasets import ImageFolder

from config import *

def main(args):

    log = ut.Logger()
    log.open(OUTPUT_FILE, mode='w')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    input_trans = transforms.Compose([
                transforms.Lambda(lambda x: ut.crop_img(x, 2000, 1300)),
                transforms.Lambda(lambda x: ut.make_square(x)),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45,PIL.Image.BILINEAR),
                transforms.ToTensor(),
                normalize
        ])

    valid_trans = transforms.Compose([
            transforms.Lambda(lambda x: ut.crop_img(x, 2000, 1300)),
            transforms.Lambda(lambda x: ut.make_square(x)),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize
    ])

    # data loader for training
    dset_train = ImageFolder(root=TRAIN_DIR, transform=input_trans)
    labels = dset_train.classes  # all lables
    num_classes = len(labels)    # number of classes
    label_map = dict({dset_train.class_to_idx[name]: name for name in dset_train.classes})

    # data loader for validating
    dset_valid = ImageFolder(root=VALID_DIR, transform=valid_trans)

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
    """
    # SGD
    optimizer = torch.optim.SGD(model.parameters(),
                                 args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    if GPU_AVAIL:
        model = model.cuda()
        criterion = criterion.cuda()
        log.write("Using GPU...\n")

    #-----------------------------Data augmentation----------------------------#
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
    #--------------------------------------------------------------------------#


    #-----------------------------Training model ------------------------------#
    model = mu.train(train_loader, valid_loader, model, criterion, optimizer, args,log)
    #--------------------------------------------------------------------------#

    #-------------------------------- Testing ---------------------------------#
    dset_test = pu.DataLoader(None, TEST_DIR, valid_trans, labels)
    test_loader = DataLoader(dset_test,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=GPU_AVAIL)
    mu.predict(test_loader, model, args, label_map, log)
    #--------------------------------------------------------------------------#

    return 0


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                if name.islower() and not name.startswith("__")
                   and callable(models.__dict__[name]))

    prs = argparse.ArgumentParser(description='Fish challenge')
    prs.add_argument('-message', default=' ', type=str, help='Message to describe experiment in spreadsheet')
    prs.add_argument('-img_size', default=224, type=int, help='image height (default: 224)')
    prs.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    prs.add_argument('-epochs', default=10, type=int, help='Number of total epochs to run')
    prs.add_argument('-lr_patience', default=3, type=int, help='Number of patience to update lr')
    prs.add_argument('-early_stop', default=5, type=int, help='Early stopping')
    prs.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='Number of data loading workers')
    prs.add_argument('-lr', '--lr', default=0.001, type=float, metavar='LR', help='Initial learning rate')
    prs.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='Mini-batch size (default: 16)')
    prs.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    prs.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    prs.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

    args = prs.parse_args()
    main(args)

    print('running correctly')

#ps.imshow()
