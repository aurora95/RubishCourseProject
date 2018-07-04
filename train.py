from __future__ import absolute_import

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from PSDataset import PSDataset
from model import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation network')
    parser.add_argument('model_name',
                        help='the name of model definition function',
                        default=None, type=str)
    parser.add_argument('exp_id', help='experiment_id', default=None, type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
    parser.add_argument('--val_batch_size', dest='val_batch_size', default=64, type=int)
    parser.add_argument('--epochs', dest='epochs', default=50, type=int)

    parser.add_argument('--base_lr', dest='base_lr', default=0.01, type=float)
    parser.add_argument('--milestones', dest='milestones', nargs=2, default=[20,30], type=int)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.0001, type=float)

    parser.add_argument('--workers', dest='workers', default=4, type=int)
    parser.add_argument('--max_queue_size', dest='max_queue_size', default=16, type=int)
    parser.add_argument('--gpu', dest='gpus',
                        nargs='*',
                        help='GPU device id to use',
                        default=[0], type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train_batch(inputs, targets, net, optimizer, loss_function, metric_functions):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    for m in metric_functions:
        m.update(preds=[outputs], labels=[targets])
    return loss

def validate(net, metric_functions, valid_dataset, valid_loader, args):
    net.eval()
    num_steps = len(valid_dataset)//args.val_batch_size + int((len(valid_dataset)%args.val_batch_size) > 0)
    for m in metric_functions:
        m.reset()
    for i, (inputs, targets) in enumerate(valid_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        for m in metric_functions:
            m.update(preds=[outputs], labels=[targets])
        progress_bar(i, num_steps, '')

def train(net, args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(os.path.join(current_dir, 'results/')) == False:
        os.mkdir(os.path.join(current_dir, 'results/'))
    save_path = 'results/'
    if os.path.exists(os.path.join(current_dir, save_path)) == False:
        os.mkdir(os.path.join(current_dir, save_path))
    save_path += '%s/'%args.model_name
    if os.path.exists(os.path.join(current_dir, save_path)) == False:
        os.mkdir(os.path.join(current_dir, save_path))
    save_path += '%s/'%args.exp_id
    if os.path.exists(os.path.join(current_dir, save_path)) == False:
        os.mkdir(os.path.join(current_dir, save_path))
    logger = Logger(save_path + 'logs/')


    train_dataset = PSDataset(
            '/home/xing/.kaggle/competitions/plant-seedlings-classification/train', 
            'train', 0.9,
            image_size=320,
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True
        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    valid_dataset = PSDataset(
            '/home/xing/.kaggle/competitions/plant-seedlings-classification/train', 
            'valid', 0.9,
            rotation_range=0,
            width_shift_range=0.,
            height_shift_range=0.,
            zoom_range=0.,
            horizontal_flip=False,
            vertical_flip=False,
        )
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.val_batch_size, num_workers=args.workers, shuffle=False)

    net = net()
    net.cuda()
    cudnn.benchmark = True
    optimizer = torch.optim.SGD(net.parameters(), lr=args.base_lr,
                                momentum=0.9, weight_decay=args.weight_decay,
                                nesterov=True)
    net = torch.nn.DataParallel(net, device_ids=args.gpus)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    loss_function = nn.CrossEntropyLoss()
    metric_functions = [Accuracy(name='Acc')]
    best_val_acc = 0.0
    num_steps = len(train_dataset)//args.batch_size + int((len(train_dataset)%args.batch_size) > 0)

    for epoch in range(args.epochs):
        scheduler.step()
        print('training epoch %d/%d, lr=%.4f:'%(epoch+1, args.epochs, optimizer.state_dict()['param_groups'][0]['lr']))
        for m in metric_functions:
            m.reset()
        train_loss = 0.
        net.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            loss = train_batch(inputs, targets, net, optimizer, loss_function, metric_functions)
            info = ''
            train_loss += loss.detach().cpu().numpy()
            info += '| loss: %.3f'%(train_loss/(i+1))
            for m in metric_functions:
                name, value = m.get()
                info += ' | %s: %.3f'%(name, value)
            progress_bar(i, num_steps, info)
        # write logs for this epoch
        logger.scalar_summary('loss', train_loss/num_steps, epoch)
        for m in metric_functions:
            name, value = m.get()
            logger.scalar_summary(name, value, epoch)
        #torch.save(net.state_dict(), save_path+'checkpoint_best.params')

        print('validating:')
        validate(net, metric_functions, valid_dataset, valid_loader, args)
        for m in metric_functions:
            name, value = m.get()
            print('--------val_{}: {}'.format(name, value))
            logger.scalar_summary('val_{}'.format(name), value, epoch)
            if name == 'Acc' and value > best_val_acc:
                torch.save(net.state_dict(), save_path+'checkpoint_best.params')
                best_val_acc = value
            else:
                torch.save(net.state_dict(), save_path+'checkpoint.params')

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    net = globals()[model_name]
    train(net, args)
