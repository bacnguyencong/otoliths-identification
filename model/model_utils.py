# Model utility functions such as loss functions, CNN building blocks etc.

import os
import numpy as np
import pandas as pd
from util.utils import AverageMeter
from  config import *

import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


import util.data_utils as pu
import util.utils as ut
from torch.utils.data import DataLoader

import os
from skimage.io import imread, imsave
from PIL import Image, ImageFont, ImageDraw
import matplotlib.patches as mpatches
import numpy as np
import ntpath
import shutil
import glob



def make_prediction_on_images(input_dir, output_dir, transforms, model, log=None):
    """
    Making predictions on the raw images (each one is bounded by a rectangle)
    :param input_dir:
    :param output_dir:
    :param transforms:
    :param model:
    :param log:
    :return:
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    # find all images from a dir
    img_list = []
    for img in glob.iglob(output_dir + '**/*.jpg', recursive=True):
        img_list.append(os.path.abspath(img))

    for img in glob.iglob(output_dir + '**/*.tif', recursive=True):
        img_list.append(os.path.abspath(img))

    font = ImageFont.truetype("FreeSerif.ttf", 30)
    cl_coding = ['green', 'cyan', 'blue', 'red', 'pink', 'orange']

    for img in img_list:
        log.write('Segmenting on {} \n'.format(img))
        image = imread(img)
        regions = ut.segment_image(image.copy(), remove_bg=False)

        # get all subfigures to PIL format
        PIL_img_list = []
        for i in range(len(regions)):
            region = regions[i]
            min_row, min_col, max_row, max_col = region.bbox
            im = image[min_row:max_row][:, min_col:max_col]

            PIL_img_list.append(Image.fromarray(im))

        log.write('Predicting on {} \n'.format(img))
        # perform testing on subfigures
        labels, probs = predict_labels(PIL_img_list, transforms, model)
        image = Image.fromarray(image)
        dr = ImageDraw.Draw(image)
        # drawing the predictions
        for i in range(len(regions)):
            minr, minc, maxr, maxc = regions[i].bbox
            color = cl_coding[labels[i]]

            for w in range(5):
                dr.rectangle(((minc + w, minr + w), (maxc - w, maxr - w)), outline=color)

            dr.text((minc, minr), str(i + 1), font=font, fill=color)
            dr.text((maxc, (minr+maxr)/2), '{:.4f}'.format(probs[i]) , font=font, fill=color)

        image.save(os.path.abspath(img))


def predict_labels(imgs, transforms, model):
    dset_test = pu.DataLoaderFromPILL(imgs,  transforms)
    test_loader = DataLoader(dset_test,
                              batch_size=4,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=GPU_AVAIL)

    return make_prediction_per_batch(test_loader, model)


def make_prediction_per_batch(data_loader, model):
    """Make prediction
    Args:
        data_loader: the loader for the data set
        model: the trained model
        log: the logger to write error messages
    Returns:
        preds: predicted labels
        probs: predicted probabilites
    """

    # switch to evaluate mode
    model.eval()
    preds, probs = [], []

    for batch_idx, inputs in enumerate(data_loader):

        input_var = Variable(inputs['image'].cuda() if GPU_AVAIL else inputs['image'])

        # forward net
        outputs = model(input_var)
        prob, pred = torch.max(outputs.data, 1)

        preds.extend(pred)
        probs.extend(prob)

    return (np.array(preds).astype(np.int), np.array(probs))


def predict(data_loader, model, args, label_map, log=None):
    """Make prediction
    Args:
        data_loader: the loader for the data set
        model: the trained model
        args: the input arguments
        label_map: the mapped labels
        log: the logger to write error messages
    """
    # switch to evaluate mode
    model.eval()
    images, preds =  [], []

    for batch_idx, sample in enumerate(data_loader):

        dinput = sample['image'].cuda() if GPU_AVAIL else sample['image']
        input_var = Variable(dinput)

         # compute output
        output = model(input_var)
        _, pred = torch.max(output.data, 1)
        pred = pred.cpu().numpy().tolist()
        preds.extend([label_map[it] for it in pred])
        images.extend(sample['name'])

    df = pd.DataFrame()
    df['image'] = images
    df['label'] = preds

    df.to_csv(os.path.join(OUTPUT_WEIGHT_PATH, 'predictions.csv'), index=False)
    log.write('Finished predicting all images ...\n')

def accuracy(output, target):
    """ Compute the accuracy """
    _, pred = torch.max(output, 1)
    return (pred == target).double().sum()/target.size(0)


def train(train_loader, valid_loader, model, criterion, optimizer, args, log=None):
    """ Training the model
    Args:
        train_loader: the loader for training data set
        valid_loader: the loader for valid data set
        model: the model arquitechture
        optimizer: the optimization solver
        args: arguments for input
        log: Logger to write output
    """

    best_acc = -float('inf') # best accuracy
    best_loss = float('inf') # best loss
    plateau_counter = 0      # counter of plateau
    lr_patience = args.lr_patience # patience for lr scheduling
    early_stopping_patience = args.early_stop # patience for early stopping
    bestpoint_file = os.path.join(OUTPUT_WEIGHT_PATH, 'best_{}.pth.tar'.format(model.modelName))

    tr_loss, tr_acc, va_loss, va_acc = [], [], [], []

    for epoch in range(0, args.epochs):

        if plateau_counter > early_stopping_patience:
            log.write('Early stopping (patience reached)...\n')
            break

        # training the model
        loss, acc = run_epoch(train_loader, model, criterion, optimizer, epoch, args.epochs, log)
        tr_loss.append(loss)
        tr_acc.append(acc)

        # validate the valid data
        loss, acc,_, _ = evaluate(model, valid_loader, criterion)
        va_loss.append(loss)
        va_acc.append(acc)

        log.write("\nValid_loss={:.4f}, valid_acc={:.4f}\n".format(loss, acc))

        # remember best accuracy and save checkpoint
        if acc > best_acc:
            best_acc, best_loss = acc, loss
            plateau_counter = 0
            # Save only the best state. Update each time the model improves
            log.write('Saving best model architecture...\n')
            torch.save({
                'epoch': epoch + 1,
                'arch': model.modelName,
                'state_dict': model.state_dict(),
                'acc': acc,
                'loss': loss,
                'optimizer': optimizer.state_dict(),
            }, bestpoint_file)
        else:
            plateau_counter += 1

        if plateau_counter > lr_patience:
            log.write('Validation loss reached plateau. Reducing learning rate...\n')
            optimizer = adjust_lr_on_plateau(optimizer)

        log.write("----------------------------------------------------------\n")

    # load the best model
    checkpoint = torch.load(os.path.join(OUTPUT_WEIGHT_PATH, 'best_{}.pth.tar'.format(model.modelName)))
    model.load_state_dict(checkpoint['state_dict'])

    _, _, true_labels, pred_labels = evaluate(model, valid_loader, criterion)

    return model, tr_loss, tr_acc, va_loss, va_acc, true_labels, pred_labels


def run_epoch(train_loader, model, criterion, optimizer, epoch, num_epochs, log=None):
    """Run one epoch of training."""

    # switch to train mode
    model.train()

    # define loss and dice recorder
    losses = AverageMeter()
    acc = AverageMeter()

    data_size = len(train_loader.dataset)

    # number of iterations before print outputs
    print_iter = np.ceil(data_size / (10 * train_loader.batch_size))

    for batch_idx, (dinput, target) in enumerate(train_loader):

        dinput = dinput.cuda() if GPU_AVAIL else dinput
        target = target.cuda() if GPU_AVAIL else target

        input_var = Variable(dinput)
        target_var = Variable(target)

        # reset gradient
        optimizer.zero_grad()

         # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.data[0], dinput.size(0))
        acc.update(prec, dinput.size(0))

        # print all messages
        if (batch_idx + 1) % print_iter == 0:
            log.write('Epoch [{:>2}][{:>5.2f} %]\t'
                  'Loss {:.4f}\t'
                  'Acc {:.4f}\n'.format(
                   epoch + 1, batch_idx*100/len(train_loader),
                   losses.avg, acc.avg))

    return losses.avg, acc.avg


def evaluate(model, data_loader, criterion):
    """ Evaluate model on labeled data. Used for evaluating on validation data.
    Args:
        model: the trained model
        data_loader: the loader of data set
        criterion: the loss function
    Return:
        loss, accuracy
    """

    # switch to evaluate mode
    model.eval()

    # define loss and accuracy recorder
    losses = AverageMeter()
    acces = AverageMeter()
    pred_labels, true_labels = [], []
    for batch_idx,  (dinput, target) in enumerate(data_loader):

        dinput = dinput.cuda() if GPU_AVAIL else dinput
        target = target.cuda() if GPU_AVAIL else target

        input_var = Variable(dinput)
        target_var = Variable(target)

         # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        prec = accuracy(output.data, target)

        _, pred = torch.max(output.data, 1)
        pred_labels.append(pred)
        true_labels.append(target.data)

        losses.update(loss.data[0], dinput.size(0))
        acces.update(prec, dinput.size(0))

        # measure accuracy and record loss
        acces.update(prec, dinput.size(0))
        losses.update(loss.data[0], dinput.size(0))

    return losses.avg, acces.avg, true_labels, pred_labels


def adjust_lr_on_plateau(optimizer):
    """Decrease learning rate by factor 10 if validation loss reaches a plateau"""
    for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10
    return optimizer
