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


def make_prediction(data_loader, model, args, log=None):
    """Make prediction
    Args:
        data_loader: the loader for the data set
        model: the trained model
        args: the input arguments
        log: the logger to write error messages
    """
    # switch to evaluate mode
    model.eval()
    images, groups, preds =  [], [], []

    for batch_idx, inputs in enumerate(data_loader):

        input_var = Variable(inputs['image'].cuda() if GPU_AVAIL else inputs['image'])

        # forward net
        outputs = model(input_var)
        prob, pred, prob_sublevel, pred_sublevel = predict(model, outputs)

        preds.extend([model.args['idx_to_lab'][it] for it in pred_sublevel])
        groups.extend([model.args['gr_lab'][it] for it in pred])
        images.extend(inputs['name'])

    df = pd.DataFrame()
    df['image'] = images
    df['group'] = groups
    df['label'] = preds

    df.to_csv(os.path.join(OUTPUT_WEIGHT_PATH, 'predictions.csv'), index=False)
    log.write('Finished predicting all images ...\n')

def predict(model, outputs, mask=None):
    """ Perform prediction
    Args:
        mask: if the group is known in advance
    Returns:
        prob: probabilities at the group level (being group 1)
        pred: prediction at the group level (0 or 1)
        prob_sublevel: probabilies at the sublevel (accoring to each group)
        pred_sublevel: predictions at the sublevel (accoring to each group)
    """

    batch_size = outputs.size(0)

    # loss at the first level of being group 1
    prob = nn.functional.sigmoid(model.level_0(outputs)) # probabilities at first level
    pred = (prob >= 0.5).data.cpu().numpy().reshape(-1) # predictions at first level
    prob = prob.data.cpu().numpy().reshape(-1)

    #if mask is None:
    mask = pred == 1

    prob_sublevel = np.zeros((mask.shape[0], 3), np.float) # probabilities at second level
    pred_sublevel = np.zeros(mask.shape, np.int) # predictions at second level

    # loss at the second level
    if np.any(~mask):
        idx = np.where(~mask)[0].tolist()
        prob_0 = nn.functional.softmax(model.level_1_0(outputs[idx, :]), dim=1)
        _, indices = prob_0.max(1)
        pred_sublevel[~mask] = [model.args['gr_0_idx'][i] for i in indices.data]
        prob_sublevel[~mask,:] = prob_0.data.cpu().numpy()

    if np.any(mask):
        idx = np.where(mask)[0].tolist()
        prob_1 = nn.functional.softmax(model.level_1_1(outputs[idx, :]), dim=1)
        _, indices = prob_1.max(1)
        pred_sublevel[mask] = [model.args['gr_1_idx'][i] for i in indices.data]
        prob_sublevel[mask,:] = prob_1.data.cpu().numpy()

    return prob, pred, prob_sublevel, pred_sublevel

def train(train_loader, valid_loader, model, optimizer, args, log=None):
    """ Train the model
    Args:
        train_loader: the loader for training data set
        valid_loader: the loader for valid data set
        model: the model arquitechture
        optimizer: the optimization solver
        args: model arguments
        log: Logger to write output
    Returns:
        model: the trained model
    """
    # loss for the first level
    BCELoss = nn.BCEWithLogitsLoss()
    # loss for the second level
    CETLoss = nn.CrossEntropyLoss()

    if GPU_AVAIL:
        BCELoss = BCELoss.cuda()
        CETLoss = CETLoss.cuda()

    best_acc_level_0 = -float('inf') # best accuracy
    best_acc_level_1 = -float('inf') # best accuracy
    best_loss = float('inf') # best loss
    best_true_labels, best_pred_labels = [], [] # used to compute confusion matrix

    plateau_counter = 0      # counter of plateau
    lr_patience = args.lr_patience # patience for lr scheduling
    early_stopping_patience = args.early_stop # patience for early stopping
    bestpoint_file = os.path.join(OUTPUT_WEIGHT_PATH, 'best_{}.pth.tar'.format(model.modelName))

    # tracking all losses and accuracies
    # valid set
    all_valid_losses = []
    all_valid_acc_level_0 = []
    all_valid_acc_level_1 = []
    # training set
    all_train_losses = []
    all_train_acc_level_0 = []
    all_train_acc_level_1 = []

    for epoch in range(0, args.epochs):

        if plateau_counter > early_stopping_patience:
            log.write('Early stopping (patience reached)...\n')
            break

        # training the model
        loss, acc_level_0, acc_level_1 = run_epoch(train_loader, model, BCELoss, CETLoss, optimizer, epoch, args.epochs, log)
        all_train_losses.append(loss)
        all_train_acc_level_0.append(acc_level_0)
        all_train_acc_level_1.append(acc_level_1)

        # validate the valid data
        loss, acc_level_0, acc_level_1,true_labels, pred_labels = evaluate(model, BCELoss, CETLoss, valid_loader)
        all_valid_losses.append(loss)
        all_valid_acc_level_0.append(acc_level_0)
        all_valid_acc_level_1.append(acc_level_1)

        log.write("\nValid_loss={:.4f}, valid_acc_level_0={:.4f}, valid_acc_level_1={:.4f}\n" \
                .format(loss, acc_level_0, acc_level_1))

        # remember best accuracy and save checkpoint
        if acc_level_1 > best_acc_level_1:
            # saving the best results
            best_loss, best_acc_level_0, best_acc_level_1 = loss, acc_level_0, acc_level_1
            best_true_labels, best_pred_labels = true_labels, pred_labels
            plateau_counter = 0
            # Save only the best state. Update each time the model improves
            log.write('Saving best model architecture...\n')
            torch.save({
                'epoch': epoch + 1,
                'arch': model.modelName,
                'state_dict': model.state_dict(),
                'acc_level_0': acc_level_0,
                'acc_level_1': acc_level_1,
                'loss': loss,
                'optimizer': optimizer.state_dict(),
            }, bestpoint_file)
        else:
            plateau_counter += 1

        if plateau_counter > lr_patience:
            log.write('Validation loss reached plateau. Reducing learning rate...\n')
            optimizer = adjust_lr_on_plateau(optimizer)

        log.write("----------------------------------------------------------\n")

    log.write("\nValid_loss={:.4f}, valid_acc_level_0={:.4f}, valid_acc_level_1={:.4f}\n" \
                .format(best_loss, best_acc_level_0, best_acc_level_1))
    # load the best model
    checkpoint = torch.load(os.path.join(OUTPUT_WEIGHT_PATH, 'best_{}.pth.tar'.format(model.modelName)))
    model.load_state_dict(checkpoint['state_dict'])

    return model, all_train_losses, all_train_acc_level_0, all_train_acc_level_1, \
            all_valid_losses, all_valid_acc_level_0, all_valid_acc_level_1, \
            best_true_labels, best_pred_labels


def run_epoch(train_loader, model, BCELoss, CETLoss, optimizer, epoch, num_epochs, log=None):
    """Run one epoch of training."""
    # switch to train mode
    model.train()

    # define loss and accuracies at two levels
    losses = AverageMeter()
    acc_level_0 = AverageMeter()
    acc_level_1 = AverageMeter()

    data_size = len(train_loader.dataset)

    # number of iterations before print outputs
    print_iter = np.ceil(data_size / (10 * train_loader.batch_size))

    for batch_idx, (inputs, y) in enumerate(train_loader):

        y = y.numpy() # list of label indices
        targets = categorical_to_binary_tensor(model, y)
        targets = targets.cuda() if GPU_AVAIL else targets

        input_var = Variable(inputs.cuda() if GPU_AVAIL else inputs)
        target_var = Variable(targets)

        batch_size = inputs.size(0)

        # reset gradient
        optimizer.zero_grad()

        # forward net
        outputs = model(input_var)

        # variable for sub input for each group
        input_var_0, target_var_0 = input_to_tensor(model, outputs, y, np.in1d(y, model.args['gr_0_idx']))
        #print(y)
        input_var_1, target_var_1 = input_to_tensor(model, outputs, y, np.in1d(y, model.args['gr_1_idx']))

        # loss at the first level
        loss = BCELoss(model.level_0(outputs), target_var)

        # loss at the second level
        if not input_var_0 is None:
            input_var_0 = model.level_1_0(input_var_0)
            loss += CETLoss(input_var_0, target_var_0) * (target_var_0.data.size(0) / batch_size)

        if not input_var_1 is None:
            input_var_1 = model.level_1_1(input_var_1)
            loss += CETLoss(input_var_1, target_var_1) * (target_var_1.data.size(0) / batch_size)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prob, pred, prob_sublevel, pred_sublevel = predict(model, outputs)

        losses.update(loss.data[0], inputs.size(0))
        acc = (pred == np.array([model.args['gr_idx'][i] for i in y])).sum() / inputs.size(0)
        acc_level_0.update(acc, inputs.size(0))
        acc = (pred_sublevel == y).sum() / inputs.size(0)
        acc_level_1.update(acc, inputs.size(0))

        # print all messages
        if ((batch_idx + 1) % print_iter == 0) or (losses.count == data_size):
            log.write( 'Epoch [{:>2}][{:>6.2f} %]\t'
                       'Loss {:.4f}\t'
                       'acc_level_0 {:.4f}\t'
                       'acc_level_1 {:.4f}\n'.format(
                            epoch + 1, losses.count*100/data_size,
                            losses.avg, acc_level_0.avg, acc_level_1.avg))

    return losses.avg, acc_level_0.avg, acc_level_1.avg

def evaluate(model, BCELoss, CETLoss, data_loader):
    """ Evaluate model on labeled data. Used for evaluating on validation data.
    Args:
        model: the trained model
        BCELoss: the loss function at first level
        CETLoss: the loss function at second level
        data_loader: the loader of data set
    Return:
        loss, acc_level_0, acc_level_1
    """

    # switch to evaluate mode
    model.eval()

    # define loss and accuracies at two levels
    losses = AverageMeter()
    acc_level_0 = AverageMeter()
    acc_level_1 = AverageMeter()
    true_labels, pred_labels = [], []


    for batch_idx, (inputs, y) in enumerate(data_loader):

        y = y.numpy() # list of label indices
        targets = categorical_to_binary_tensor(model, y)
        targets = targets.cuda() if GPU_AVAIL else targets

        input_var = Variable(inputs.cuda() if GPU_AVAIL else inputs, requires_grad=False)
        target_var = Variable(targets, requires_grad=False)

        batch_size = inputs.size(0)

        # forward net
        outputs = model(input_var)

        # variable for sub input for each group
        input_var_0, target_var_0 = input_to_tensor(model, outputs, y, np.in1d(y, model.args['gr_0_idx']))
        input_var_1, target_var_1 = input_to_tensor(model, outputs, y, np.in1d(y, model.args['gr_1_idx']))

        # loss at the first level
        loss = BCELoss(model.level_0(outputs), target_var)

        # loss at the second level
        if not input_var_0 is None:
            input_var_0 = model.level_1_0(input_var_0)
            loss += CETLoss(input_var_0, target_var_0) * (target_var_0.data.size(0) / batch_size)

        if not input_var_1 is None:
            input_var_1 = model.level_1_1(input_var_1)
            loss += CETLoss(input_var_1, target_var_1) * (target_var_1.data.size(0) / batch_size)

        # measure accuracy and record loss
        _, pred, _, pred_sublevel = predict(model, outputs)

        losses.update(loss.data[0], inputs.size(0))
        acc = (pred == np.array([model.args['gr_idx'][i] for i in y])).sum() / inputs.size(0)
        acc_level_0.update(acc, inputs.size(0))
        acc = (pred_sublevel == y).sum() / inputs.size(0)
        acc_level_1.update(acc, inputs.size(0))

        pred_labels.extend(pred_sublevel)
        true_labels.extend(y.tolist())

    return losses.avg, acc_level_0.avg, acc_level_1.avg, true_labels, pred_labels


def adjust_lr_on_plateau(optimizer):
    """Decrease learning rate by factor 10 if validation loss reaches a plateau"""
    for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10
    return optimizer


def categorical_to_binary_tensor(model, labels):
    """ return a tensor 0 if it belongs to group 0, and 1 if it belongs to group 1"""
    mask = np.in1d(labels, model.args['gr_1_idx']).astype(np.int)
    return torch.from_numpy(mask.reshape(-1,1)).float()

def input_to_tensor(model, inputs, labels, mask):
    # check if exists the input
    if np.any(mask):
        index = np.where(mask)[0].tolist()
        y = [ model.args['idx_to_subidx'][it] for it in labels[mask].flatten() ]
        y = torch.from_numpy(np.array(y).reshape(-1)).long()
        y = y.cuda() if GPU_AVAIL else y

        return inputs[index, :], Variable(y)

    return None, None
