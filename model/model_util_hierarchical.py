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


def train(train_loader, valid_loader, model, optimizer, args, log=None):
    """ Training the model
    Args:
        train_loader: the loader for training data set
        valid_loader: the loader for valid data set
        model: the model arquitechture
        optimizer: the optimization solver
        args: arguments for input
        log: Logger to write output
    """
    # loss for the first level
    BCELoss = nn.BCEWithLogitsLoss()
    # loss for the second level
    CETLoss = nn.CrossEntropyLoss()

    if GPU_AVAIL:
        BCELoss = BCELoss.cuda()
        CETLoss = CETLoss.cuda()

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
        run_epoch(train_loader, model, BCELoss, CETLoss, optimizer, epoch, args.epochs, log)

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


def run_epoch(train_loader, model, BCELoss, CETLoss, optimizer, epoch, num_epochs, log=None):
    """Run one epoch of training."""

    def categorical_to_binary_tensor(labels):
        """ return a tensor 0 if it belongs to group 0, and 1 if it belongs to group 1"""
        mask = np.in1d(labels, model.args['gr_1_idx']).astype(np.int)
        return torch.from_numpy(mask.reshape(-1,1)).float()

    def input_to_tensor(inputs, labels, mask):
        # check if exists the input
        if np.any(mask):
            index = np.where(mask)[0].tolist()
            y = [ model.args['idx_to_subidx'][it] for it in labels[mask].flatten() ]
            y = torch.from_numpy(np.array(y).reshape(-1)).long()
            y = y.cuda() if GPU_AVAIL else y

            return inputs[index, :], Variable(y)

        return None

    # switch to train mode
    model.train()

    # define loss and dice recorder
    losses = AverageMeter()
    acc = AverageMeter()

    data_size = len(train_loader.dataset)

    # number of iterations before print outputs
    print_iter = np.ceil(data_size / (10 * train_loader.batch_size))

    for batch_idx, (inputs, y) in enumerate(train_loader):
        targets = categorical_to_binary_tensor(y.data.numpy())
        targets = target.cuda() if GPU_AVAIL else targets

        input_var = Variable(inputs.cuda() if GPU_AVAIL else inputs)
        target_var = Variable(targets)

        batch_size = inputs.size(0)

        # reset gradient
        optimizer.zero_grad()

        # forward net
        outputs = model(input_var)

        # variable for sub input for each group
        input_var_0, target_var_0 = input_to_tensor(outputs, y, np.in1d(y, gr_0_idx))
        input_var_1, target_var_1 = input_to_tensor(outputs, y, np.in1d(y, gr_1_idx))

        # loss at the first level
        loss = BCELoss(net.level_0(outputs), target_var)

        # loss at the second level
        if not input_var_0 is None:
            input_var_0 = net.level_1_0(input_var_0)
            loss += CETLoss(input_var_0, target_var_0) * (target_var_0.data.size(0) / batch_size)
        if not input_var_1 is None:
            input_var_1 = net.level_1_1(input_var_1)
            loss += CETLoss(input_var_1, target_var_1) * (target_var_1.data.size(0) / batch_size)

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

    for batch_idx,  (dinput, target) in enumerate(data_loader):

        dinput = dinput.cuda() if GPU_AVAIL else dinput
        target = target.cuda() if GPU_AVAIL else target

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
