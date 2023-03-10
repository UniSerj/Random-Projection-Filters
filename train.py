import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.wide_resnet import WideResNet32
from model.resnet import ResNet18, ResNet50

from utils import evaluate_standard, evaluate_standard_rp, evaluate_pgd, evaluate_pgd_rp

from utils import clamp, get_loaders, get_limit


logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--network', default='ResNet18', type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep', 'cosine'])
    parser.add_argument('--lr_min', default=0., type=float)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=float, help='Step size')
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--device', type=str, default='0')  # gpu

    parser.add_argument('--attack_iters', default=10, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)

    parser.add_argument('--adv_training', action='store_true', help='if adv training')

    # random setting
    parser.add_argument('--rp', action='store_true', help='if random projection')
    parser.add_argument('--rp_block', default=None, type=int, nargs='*',
                        help='block schedule of rp')
    parser.add_argument('--rp_out_channel', default=0, type=int, help='number of rp output channels')
    parser.add_argument('--rp_weight_decay', default=5e-4, type=float)

    return parser.parse_args()


def main():
    args = get_args()

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    # saving path
    path = os.path.join('ckpt', args.dataset, args.network)
    args.save_dir = os.path.join(path, args.save_dir)

    # logger
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logfile = os.path.join(args.save_dir, 'output.log')
    if os.path.exists(logfile):
        try:
            os.remove(logfile)
        except:
            print('Cannot find logfile')
    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    logger.info(args)

    # set current device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    ngpus = torch.cuda.device_count()
    logger.info('Devices: [{:d}]'.format(ngpus))

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get data loader
    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker)

    # adv training attack setting
    std, upper_limit, lower_limit = get_limit(args.dataset)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    # setup network
    if args.network == 'ResNet18':
        net = ResNet18
    elif args.network == 'ResNet50':
        net = ResNet50
    elif args.network == 'WideResNet32':
        net = WideResNet32
    else:
        print('Wrong network:', args.network)

    model = net(num_classes=args.num_classes, rp=args.rp, rp_block=args.rp_block, rp_out_channel=args.rp_out_channel,
                normalize=dataset_normalization).cuda()

    model = torch.nn.DataParallel(model)
    logger.info(model)

    # set weight decay for random projection layer
    params = []
    for name, param in model.named_parameters():
        if 'rp_conv' in name:
            params.append({'params': param, 'weight_decay': args.rp_weight_decay})
        else:
            params.append({'params': param})

    # setup optimizer, loss function, LR scheduler
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if args.lr_schedule == 'cyclic':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, float(args.epochs))

    best_pgd_acc = 0
    best_clean_acc = 0
    test_acc_best_pgd = 0

    # load pretrained model
    if os.path.exists(os.path.join(args.save_dir, 'model_latest.pth')):
        pretrained_model = torch.load(os.path.join(args.save_dir, 'model_latest.pth'), map_location='cpu')
        partial = pretrained_model['state_dict']

        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
        try:
            opt.load_state_dict(pretrained_model['opt'])
        except:
            print("Cannot load optimization")
        scheduler.load_state_dict(pretrained_model['scheduler'])
        start_epoch = pretrained_model['epoch'] + 1

        best_pgd_acc = pretrained_model['best_pgd_acc']
        test_acc_best_pgd = pretrained_model['standard_acc']

        print('Resume from Epoch %d. Load pretrained weight.' % start_epoch)

    else:
        start_epoch = 0
        print('No checkpoint. Train from scratch.')

    # Start training
    start_train_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.train()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            _iters = epoch * len(train_loader) + i

            # random select a path to attack
            if args.rp:
                model.module.random_rp_matrix()

            X, y = X.to('cuda'), y.to('cuda')
            if args.adv_training:
                # init delta
                delta = torch.zeros_like(X).to('cuda')
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True

                # pgd attack
                for _ in range(args.attack_iters):
                    output = model(X + delta)
                    loss = criterion(output, y)

                    loss.backward()

                    grad = delta.grad.detach()

                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.grad.zero_()

                delta = delta.detach()

                # random select a path to infer
                if args.rp:
                    model.module.random_rp_matrix()

                output = model(X + delta[:X.size(0)])
            else:
                output = model(X)

            loss = criterion(output, y)

            opt.zero_grad()
            loss.backward()

            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            if i % 150 == 0:
                logger.info("Device: [{:s}]\t"
                            "Iter: [{:d}][{:d}/{:d}]\t"
                            "Loss {:.3f} ({:.3f})\t"
                            "Prec@1 {:.3f} ({:.3f})\t".format(
                    args.device,
                    epoch,
                    i,
                    len(train_loader),
                    loss.item(),
                    train_loss / train_n,
                    (output.max(1)[1] == y).sum().item() / y.size(0),
                    train_acc / train_n)
                )

        if args.rp:
            logger.info('Evaluating with standard images with random projection...')
            test_loss, test_acc = evaluate_standard_rp(test_loader, model)
            logger.info('Evaluating with PGD Attack with random projection...')
            if args.adv_training:
                pgd_loss, pgd_acc = evaluate_pgd_rp(test_loader, model, 20, 1, args, num_round=3)
            else:
                pgd_loss, pgd_acc = 0.0, 0.0
        else:
            logger.info('Evaluating with standard images...')
            test_loss, test_acc = evaluate_standard(test_loader, model)
            logger.info('Evaluating with PGD Attack...')
            if args.adv_training:
                pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 20, 1, args)
            else:
                pgd_loss, pgd_acc = 0.0, 0.0


        if pgd_acc > best_pgd_acc and args.adv_training:
            best_pgd_acc = pgd_acc
            test_acc_best_pgd = test_acc

            best_state = {}
            best_state['state_dict'] = copy.deepcopy(model.state_dict())
            best_state['opt'] = copy.deepcopy(opt.state_dict())
            best_state['scheduler'] = copy.deepcopy(scheduler.state_dict())
            best_state['epoch'] = epoch
            best_state['best_pgd_acc'] = best_pgd_acc
            best_state['standard_acc'] = test_acc

            torch.save(best_state, os.path.join(args.save_dir, 'model.pth'))
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'weight.pth'))
        elif test_acc > best_clean_acc and not args.adv_training:
            best_clean_acc = test_acc
            best_state = {}
            best_state['state_dict'] = copy.deepcopy(model.state_dict())
            best_state['opt'] = copy.deepcopy(opt.state_dict())
            best_state['scheduler'] = copy.deepcopy(scheduler.state_dict())
            best_state['epoch'] = epoch
            best_state['standard_acc'] = test_acc

            torch.save(best_state, os.path.join(args.save_dir, 'model.pth'))
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'weight.pth'))

        # save latest ckpt
        if args.adv_training:
            best_state = {}
            best_state['state_dict'] = copy.deepcopy(model.state_dict())
            best_state['opt'] = copy.deepcopy(opt.state_dict())
            best_state['scheduler'] = copy.deepcopy(scheduler.state_dict())
            best_state['epoch'] = epoch
            best_state['best_pgd_acc'] = best_pgd_acc
            best_state['standard_acc'] = test_acc_best_pgd

            torch.save(best_state, os.path.join(args.save_dir, 'model_latest.pth'))
        else:
            best_state = {}
            best_state['state_dict'] = copy.deepcopy(model.state_dict())
            best_state['opt'] = copy.deepcopy(opt.state_dict())
            best_state['scheduler'] = copy.deepcopy(scheduler.state_dict())
            best_state['epoch'] = epoch
            best_state['standard_acc'] = best_clean_acc

            torch.save(best_state, os.path.join(args.save_dir, 'model_latest.pth'))

        logger.info(
            "Device: [{:s}]\t"
            "Test Loss: {:.4f}\t"
            "Test Acc: {:.4f}\n"
            "PGD Loss: {:.4f}\t"
            "PGD Acc: {:.4f}\n"
            "Best PGD Acc: {:.4f}\t"
            "Test Acc of best PGD ckpt: {:.4f}".format(
                args.device, test_loss, test_acc, pgd_loss, pgd_acc, best_pgd_acc, test_acc_best_pgd)
        )


    train_time = time.time()
    logger.info('Total train time: {:.4f} minutes'.format((train_time - start_train_time) / 60))


if __name__ == "__main__":
    main()
