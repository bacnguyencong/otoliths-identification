import util.data_utils as pu
import util.utils as ut
from torch.utils.data import DataLoader
import argparse
import torch
from torchvision import models, transforms
from model.CNNs import FineTuneModel

ROOT_DIR = './data/images'
CSV_FILE = './data/data.csv'
OUTPUT_FILE = './output/log.txt'
GPU_AVAIL = torch.cuda.is_available()

def main(args):
    
    log = ut.Logger()
    log.open(OUTPUT_FILE, mode='w')
    
    transform = None
    dset_train = pu.DataLoader(CSV_FILE, ROOT_DIR, transform)
    
    model_name = 'resnet'
    num_classes = 2
    
    #----------------------------Configure-------------------------#
    # model arquitechture
    model = FineTuneModel(models.resnet18(pretrained=True), model_name, num_classes)
    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    if GPU_AVAIL:
        model = model.cuda()
        criterion = criterion.cuda()
        log.write("Using GPU...")
    
    #----------------------------Data augmentation-------------------------#
    tloader = DataLoader(dset_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=GPU_AVAIL)
    
        
    return 0

if __name__ == '__main__':
    
    prs = argparse.ArgumentParser(description='Kaggle: Carvana car segmentation challenge')
    prs.add_argument('-message', default=' ', type=str, help='Message to describe experiment in spreadsheet')
    prs.add_argument('-im_h', default=1280, type=int, help='image height (default: 1280)')
    prs.add_argument('-im_w', default=1918, type=int, help='image width (default: 1918)')
    prs.add_argument('-arch', default='resnet', help='Model architecture')
    prs.add_argument('-epochs', default=100, type=int, help='Number of total epochs to run')
    prs.add_argument('-j', '--workers', default=3, type=int, metavar='N', help='Number of data loading workers')
    prs.add_argument('-lr', '--lr', default=0.01, type=float, metavar='LR', help='Initial learning rate')
    prs.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='Mini-batch size (default: 16)')
    prs.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    
    args = prs.parse_args()  
    
    main(args)
    print('running correctly')

#ps.imshow()