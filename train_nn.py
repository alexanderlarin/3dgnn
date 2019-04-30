import os
import sys
import time
import logging

import cv2
import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook

import config
from datasets import nyudv2
from models import Model

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

logger = logging.getLogger('3dgnn')

LABEL_IDX = {'<UNK>': 0, 'beam': 1, 'board': 2, 'bookcase': 3, 'ceiling': 4, 'chair': 5, 'clutter': 6,
             'column': 7, 'door': 8, 'floor': 9, 'sofa': 10, 'table': 11, 'wall': 12, 'window': 13}

IDX_LABEL = {0: '<UNK>', 1: 'beam', 2: 'board', 3: 'bookcase', 4: 'ceiling', 5: 'chair', 6: 'clutter',
             7: 'column', 8: 'door', 9: 'floor', 10: 'sofa', 11: 'table', 12: 'wall', 13: 'window'}


def train_nn(dataset_path, hha_dir, save_models_dir, num_epochs=50, batch_size=4,
             start_epoch=1, pre_train_model='', notebook=False):
    progress = tqdm_notebook if notebook else tqdm
    logger.info('Loading data...')

    dataset_tr = nyudv2.Dataset(dataset_path, hha_dir, flip_prob=config.flip_prob, crop_type='Random', crop_size=config.crop_size)
    dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True,
                               num_workers=config.workers_tr, drop_last=False, pin_memory=True)

    dataset_va = nyudv2.Dataset(dataset_path, hha_dir, flip_prob=0.0, crop_type='Center', crop_size=config.crop_size)
    dataloader_va = DataLoader(dataset_va, batch_size=batch_size, shuffle=False,
                               num_workers=config.workers_va, drop_last=False, pin_memory=True)

    cv2.setNumThreads(config.workers_tr)

    logger.info('Preparing model...')
    model = Model(config.nclasses, config.mlp_num_layers, config.use_gpu)
    loss = nn.NLLLoss(reduce=not config.use_bootstrap_loss, weight=torch.FloatTensor(config.class_weights))
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)

    if config.use_gpu:
        model = model.cuda()
        loss = loss.cuda()
        softmax = softmax.cuda()
        log_softmax = log_softmax.cuda()

    optimizer = torch.optim.Adam([{'params': model.decoder.parameters()},
                                  {'params': model.gnn.parameters(), 'lr': config.gnn_initial_lr}],
                                 lr=config.base_initial_lr, betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay)

    if config.lr_schedule_type == 'exp':
        def lambda_1(lambda_epoch):
            return pow((1 - ((lambda_epoch - 1) / num_epochs)), config.lr_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_1)
    elif config.lr_schedule_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.lr_decay,
                                                               patience=config.lr_patience)
    else:
        logger.error('Bad scheduler')
        exit(1)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Number of trainable parameters: %d", params)

    def get_current_learning_rates():
        learning_rates = []
        for param_group in optimizer.param_groups:
            learning_rates.append(param_group['lr'])
        return learning_rates

    def eval_set(dataloader):
        model.eval()

        with torch.no_grad():
            loss_sum = 0.0
            init_tensor_value = np.zeros(14 ** 2)
            if config.use_gpu:
                confusion_matrix = torch.cuda.FloatTensor(init_tensor_value)
            else:
                confusion_matrix = torch.FloatTensor(init_tensor_value)

            start_time = time.time()

            for batch_idx, rgbd_label_xy in progress(enumerate(dataloader), total=len(dataloader), desc=f'Eval set'):
                x = rgbd_label_xy[0]
                xy = rgbd_label_xy[2]
                target = rgbd_label_xy[1].long()
                x = x.float()
                xy = xy.float()

                input = x.permute(0, 3, 1, 2).contiguous()
                xy = xy.permute(0, 3, 1, 2).contiguous()
                if config.use_gpu:
                    input = input.cuda()
                    xy = xy.cuda()
                    target = target.cuda()

                output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy,
                               use_gnn=config.use_gnn)

                if config.use_bootstrap_loss:
                    loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                    topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                               int((config.crop_size ** 2) * config.bootstrap_rate))
                    loss_ = torch.mean(topk)
                else:
                    loss_ = loss.forward(log_softmax(output.float()), target)
                loss_sum += loss_

                pred = output.permute(0, 2, 3, 1).contiguous()
                pred = pred.view(-1, config.nclasses)
                pred = softmax(pred)
                pred_max_val, pred_arg_max = pred.max(1)

                pairs = target.view(-1) * 14 + pred_arg_max.view(-1)
                for i in range(14 ** 2):
                    cumu = pairs.eq(i).float().sum()
                    confusion_matrix[i] += cumu.item()

            sys.stdout.write(" - Eval time: {:.2f}s \n".format(time.time() - start_time))
            loss_sum /= len(dataloader)

            confusion_matrix = confusion_matrix.cpu().numpy().reshape((14, 14))
            class_iou = np.zeros(14)
            confusion_matrix[0, :] = np.zeros(14)
            confusion_matrix[:, 0] = np.zeros(14)
            for i in range(1, 14):
                class_iou[i] = confusion_matrix[i, i] / (
                        np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i])

        return loss_sum.item(), class_iou, confusion_matrix

    # Training parameter
    logger.info(f'Num_epochs: {num_epochs}')
    interval_to_show = 100

    train_losses = []
    eval_losses = []

    if pre_train_model:
        logger.info('Loading pre-train model...')
        model.load_state_dict(torch.load(pre_train_model))
    else:
        logger.info('Starting training from scratch...')

    # Training
    for epoch in progress(range(start_epoch, num_epochs + 1), desc='Training'):
        batch_loss_avg = 0
        if config.lr_schedule_type == 'exp':
            scheduler.step(epoch)
        for batch_idx, rgbd_label_xy in progress(enumerate(dataloader_tr), total=len(dataloader_tr),
                                                 desc=f'Epoch {epoch}'):
            x = rgbd_label_xy[0]
            target = rgbd_label_xy[1].long()
            xy = rgbd_label_xy[2]
            x = x.float()
            xy = xy.float()

            input = x.permute(0, 3, 1, 2).contiguous()
            input = input.type(torch.FloatTensor)

            if config.use_gpu:
                input = input.cuda()
                xy = xy.cuda()
                target = target.cuda()

            xy = xy.permute(0, 3, 1, 2).contiguous()

            optimizer.zero_grad()
            model.train()

            output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy, use_gnn=config.use_gnn)

            if config.use_bootstrap_loss:
                loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                           int((config.crop_size ** 2) * config.bootstrap_rate))
                loss_ = torch.mean(topk)
            else:
                loss_ = loss.forward(log_softmax(output.float()), target)

            loss_.backward()
            optimizer.step()

            batch_loss_avg += loss_.item()

            if batch_idx % interval_to_show == 0 and batch_idx > 0:
                batch_loss_avg /= interval_to_show
                train_losses.append(batch_loss_avg)
                logger.info("E%dB%d Batch loss average: %s", epoch, batch_idx, batch_loss_avg)
                print('\rEpoch:{}, Batch:{}, loss average:{}'.format(epoch, batch_idx, batch_loss_avg))
                batch_loss_avg = 0

        batch_idx = len(dataloader_tr)
        logger.info("E%dB%d Saving model...", epoch, batch_idx)

        torch.save(model.state_dict(), os.path.join(save_models_dir, f'checkpoint_{epoch!s}.pth'))

        # Evaluation
        eval_loss, class_iou, confusion_matrix = eval_set(dataloader_va)
        eval_losses.append(eval_loss)

        if config.lr_schedule_type == 'plateau':
            scheduler.step(eval_loss)
        print('Learning ...')
        logger.info("E%dB%d Def learning rate: %s", epoch, batch_idx, get_current_learning_rates()[0])
        print('Epoch{} Def learning rate: {}'.format(epoch, get_current_learning_rates()[0]))
        logger.info("E%dB%d GNN learning rate: %s", epoch, batch_idx, get_current_learning_rates()[1])
        print('Epoch{} GNN learning rate: {}'.format(epoch, get_current_learning_rates()[1]))
        logger.info("E%dB%d Eval loss: %s", epoch, batch_idx, eval_loss)
        print('Epoch{} Eval loss: {}'.format(epoch, eval_loss))
        logger.info("E%dB%d Class IoU:", epoch, batch_idx)
        print('Epoch{} Class IoU:'.format(epoch))
        for cl in range(14):
            logger.info("%+10s: %-10s" % (IDX_LABEL[cl], class_iou[cl]))
            print('{}:{}'.format(IDX_LABEL[cl], class_iou[cl]))
        logger.info("Mean IoU: %s", np.mean(class_iou[1:]))
        print("Mean IoU: %.2f" % np.mean(class_iou[1:]))
        logger.info("E%dB%d Confusion matrix:", epoch, batch_idx)
        logger.info(confusion_matrix)

    logger.info('Finished training!')
    logger.info('Saving trained model...')
    torch.save(model.state_dict(), os.path.join(save_models_dir, 'finish.pth'))
    eval_loss, class_iou, confusion_matrix = eval_set(dataloader_va)
    logger.info('Eval loss: %s', eval_loss)
    logger.info('Class IoU:')
    for cl in range(14):
        logger.info("%+10s: %-10s" % (IDX_LABEL[cl], class_iou[cl]))
    logger.info(f'Mean IoU: {np.mean(class_iou[1:])}')
