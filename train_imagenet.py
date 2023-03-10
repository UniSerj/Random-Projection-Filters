# This module is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from utils_imagenet import *
from validation import validate, validate_pgd, validate_pgd_random, validate_random
import torchvision.models as models
from model.resnet_imagenet import ResNet50


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default='./data/val', help='path to dataset')
    parser.add_argument('-c', '--config', default='config.yml', type=str, metavar='Path',
                        help='path to the config file (default: config.yml)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--eval_model_path', default=None, type=str, help='path to the RPF model')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_schedule', default='cosine',
                        choices=['multistep', 'cosine'])
    parser.add_argument('--epochs', default=90, type=int,
                        help='total training epochs')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='training batch size')
    parser.add_argument('--clip_eps', default=4, type=float,
                        help='epsilon')
    parser.add_argument('--fgsm_step', default=4, type=float,
                        help='step size (alpha)')
    parser.add_argument('--save_dir', default=None, type=str, help='Output directory')
    parser.add_argument('--adv_train', action='store_true', help='if adv_train')

    # random setting
    parser.add_argument('--rp', action='store_true', help='if random projection')
    parser.add_argument('--rp_out_channel', default=0, type=int, help='number of rp output channels')
    parser.add_argument('--rp_weight_decay', default=5e-4, type=float)

    return parser.parse_args()


config = parse_config_file(parse_args())
if config.save_dir is not None:
    config.output_name = config.save_dir
if config.lr is not None:
    config.TRAIN.lr = config.lr
if config.epochs is not None:
    config.TRAIN.epochs = config.epochs
if config.batch_size is not None:
    config.DATA.batch_size = config.batch_size
if config.clip_eps is not None:
    config.ADV.clip_eps = config.clip_eps
if config.fgsm_step is not None:
    config.ADV.fgsm_step = config.fgsm_step


def main():
    # Parase config file and initiate logging
    logger = initiate_logger(config.output_name)
    # print = logger.info
    cudnn.benchmark = True

    # Scale and initialize the parameters
    best_pgd_acc = 0
    best_epoch = 0
    standard_acc_at_best_pgd = 0
    best_pgd_acc_k50 = 0

    config.ADV.n_repeats = 1

    config.TRAIN.epochs = int(math.ceil(config.TRAIN.epochs / config.ADV.n_repeats))
    config.ADV.fgsm_step /= config.DATA.max_color_value
    config.ADV.clip_eps /= config.DATA.max_color_value

    # Create output folder
    if not os.path.isdir(config.output_name):
        os.makedirs(config.output_name)

    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in config.items(): logger.info('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    if os.path.exists(os.path.join(config.output_name, 'model_latest.pth')):
        config.pretrained = False

    model = ResNet50(pretrained=config.pretrained, rp=config.rp, rp_out_channel=config.rp_out_channel).cuda()
    logger.info(model)

    # Wrap the model into DataParallel
    model = torch.nn.DataParallel(model)

    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), config.TRAIN.lr,
                                momentum=config.TRAIN.momentum,
                                weight_decay=config.TRAIN.weight_decay)

    # scheduler
    if config.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.epochs))

    if os.path.exists(os.path.join(config.output_name, 'model_latest.pth')):
        model_path = os.path.join(config.output_name, 'model_latest.pth')

        checkpoint = torch.load(model_path)
        config.TRAIN.start_epoch = checkpoint['epoch'] + 1
        best_pgd_acc = checkpoint['best_pgd_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> Automatic resume from '{}' (epoch {})"
                    .format(model_path, checkpoint['epoch']))
    else:
        logger.info("Train from scratch")

    # Initiate data loaders
    # traindir = os.path.join(config.data, 'train')
    traindir = os.path.join(config.data, 'val')
    valdir = os.path.join(config.data, 'val')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(config.DATA.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.DATA.batch_size, shuffle=True,
        num_workers=config.DATA.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(config.DATA.img_size),
            transforms.CenterCrop(config.DATA.crop_size),
            transforms.ToTensor(),
        ])),
        batch_size=config.DATA.batch_size, shuffle=False,
        num_workers=config.DATA.workers, pin_memory=True)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if config.evaluate:
        # load model
        if config.eval_model_path is None:
            print('Eval_model_path not defined')
            exit()
        else:
            model_path = config.eval_model_path
            checkpoint = torch.load(model_path)
            config.TRAIN.start_epoch = checkpoint['epoch'] + 1
            best_pgd_acc = checkpoint['best_pgd_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> Automatic resume from '{}' (epoch {})"
                        .format(model_path, checkpoint['epoch']))

        logger.info(pad_str(' Performing PGD Attacks '))
        validate_random(val_loader, model, criterion, config, logger)

        for pgd_param in config.ADV.pgd_attack:
            validate_pgd_random(val_loader, model, criterion, pgd_param[0], pgd_param[1], config, logger)
        return

    for epoch in range(config.TRAIN.start_epoch, config.TRAIN.epochs):
        if config.lr_schedule == 'cosine':
            scheduler.step()
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        else:
            adjust_learning_rate(config.TRAIN.lr, optimizer, epoch, config.ADV.n_repeats)

        if config.adv_train:
            train_pgd(train_loader, model, criterion, optimizer, epoch, logger)
        else:
            train(train_loader, model, criterion, optimizer, epoch, logger)


        # evaluate on validation set
        if epoch % 2 == 0:
            if config.adv_train:
                if config.rp:
                    pgd_acc = validate_pgd_random(val_loader, model, criterion, 10, 1.0 / 255.0, config, logger)
                    standard_acc = validate_random(val_loader, model, criterion, config, logger)
                else:
                    pgd_acc = validate_pgd(val_loader, model, criterion, 10, 1.0 / 255.0, config, logger)
                    standard_acc = validate(val_loader, model, criterion, config, logger)
            else:
                if config.rp:
                    pgd_acc = validate_random(val_loader, model, criterion, config, logger)
                    standard_acc = pgd_acc
                else:
                    pgd_acc = validate(val_loader, model, criterion, config, logger)
                    standard_acc = pgd_acc

            # remember best prec@1 and save checkpoint
            is_best = pgd_acc > best_pgd_acc
            best_pgd_acc = max(pgd_acc, best_pgd_acc)

            if is_best:
                best_epoch = epoch
                standard_acc_at_best_pgd = standard_acc
                if config.adv_train:
                    best_pgd_acc_k50 = validate_pgd_random(val_loader, model, criterion, 50, 1.0 / 255.0, config, logger)
                else:
                    best_pgd_acc_k50 = 0.0
            save_checkpoint({
                'epoch': epoch,
                'arch': config.TRAIN.arch,
                'state_dict': model.state_dict(),
                'best_pgd_acc': best_pgd_acc,
                'standard_acc': standard_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, config.output_name)

            # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            logger.info('Current Best PGD Acc: {:.3f}, Achieved at [{:d}] Epoch, with Standard Acc: {:.3f},'
                        ' with K50 Acc: {:.3f}'.format(best_pgd_acc, best_epoch, standard_acc_at_best_pgd, best_pgd_acc_k50))

    # Automatically perform PGD Attacks at the end of training
    logger.info(pad_str(' Performing PGD Attacks '))

    for pgd_param in config.ADV.pgd_attack:
        validate_pgd_random(val_loader, model, criterion, pgd_param[0], pgd_param[1], config, logger)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        _iters = epoch * len(train_loader) + i

        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)


        # random select a path
        if config.rp:
            model.module.random_rp_matrix()

        in1 = input
        in1.clamp_(0, 1.0)
        in1.sub_(mean).div_(std)
        output = model(in1)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.TRAIN.print_freq == 0:
            logger.info('Train Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, top1=top1, top5=top5, cls_loss=losses))
            sys.stdout.flush()


def train_pgd(train_loader, model, criterion, optimizer, epoch, logger):
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        _iters = epoch * len(train_loader) + i

        # random select a path to attack
        if config.rp:
            model.module.random_rp_matrix()

        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        delta = torch.zeros_like(input).cuda()
        if config.ADV.delta_init == 'random':
            delta.uniform_(-config.ADV.clip_eps, config.ADV.clip_eps)
        delta.requires_grad = True

        for _ in range(config.ADV.attack_iters):
            in1 = input + delta
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = model(in1)

            loss = criterion(output, target)
            loss.backward()

            grad = delta.grad.detach()
            delta.data = delta.data + config.ADV.fgsm_step * torch.sign(grad)
            delta.data.clamp_(-config.ADV.clip_eps, config.ADV.clip_eps)
            delta.grad.zero_()

        delta = delta.detach()

        # random select a path to infer
        if config.rp:
            model.module.random_rp_matrix()

        in1 = input + delta
        in1.clamp_(0, 1.0)
        in1.sub_(mean).div_(std)
        output = model(in1)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.TRAIN.print_freq == 0:
            logger.info('Train Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, top1=top1, top5=top5, cls_loss=losses))
            sys.stdout.flush()


if __name__ == '__main__':
    main()