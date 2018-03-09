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


def predict(data_loader, model, args, encoder, log=None):
    """Make prediction
    Args:
        data_loader: the loader for the data set
        model: the trained model
        args: the input arguments
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
        preds.extend(encoder.inverse_transform(pred))
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

    for epoch in range(0, args.epochs):

        if plateau_counter > early_stopping_patience:
            log.write('Early stopping (patience reached)...\n')
            break

        # training the model
        run_epoch(train_loader, model, criterion, optimizer, epoch, args.epochs, log)

        # validate the valid data
        loss, acc = evaluate(model, valid_loader, criterion)
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

    return model


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

    for batch_idx, sample in enumerate(train_loader):

        dinput = sample['image'].cuda() if GPU_AVAIL else sample['image']
        target = sample['label'].cuda() if GPU_AVAIL else sample['label']

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

    for batch_idx, sample in enumerate(data_loader):

        dinput = sample['image'].cuda() if GPU_AVAIL else sample['image']
        target = sample['label'].cuda() if GPU_AVAIL else sample['label']

        input_var = Variable(dinput)
        target_var = Variable(target)

         # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        prec = accuracy(output.data, target)
        losses.update(loss.data[0], dinput.size(0))
        acces.update(prec, dinput.size(0))

        # measure accuracy and record loss
        acces.update(prec, dinput.size(0))
        losses.update(loss.data[0], dinput.size(0))

    return losses.avg, acces.avg


def adjust_lr_on_plateau(optimizer):
    """Decrease learning rate by factor 10 if validation loss reaches a plateau"""
    for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10
    return optimizer
