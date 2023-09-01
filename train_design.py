import torch
import numpy as np
import random
import argparse
import os
import os.path as osp
import sys
import time
import json,glob
from mmcv import Config
#import mmcv.Config as Config
from dataset import build_data_loader
from models import build_model
from utils import AverageMeter,ResultFormat
from torch.optim.lr_scheduler import CosineAnnealingLR
from eval.ic15.cal_recall import cal_recall_precison_f1
from tqdm import tqdm
torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)
# Inside my model training code
# import wandb

train_step = 0

def train(train_loader, model, optimizer, epoch, start_iter, cfg,):
    model.train()
    train_loss = 0.
    global train_step

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()

    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            sys.stdout.flush()
            continue
        train_step += 1

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data.update(dict(cfg=cfg))

        # forward
        outputs = model(**data)
        #
        # print(outputs['loss_text'].shape)
        # print(outputs['loss_kernels'].shape)

        # detection loss
        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item())

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item())

        loss = loss_text + loss_kernels

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item())
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item())
        # 总的损失
        train_loss += loss.item()
        losses.update(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()


        # print log
        if iter % 20 == 0:
            output_log = '({batch}/{size}) LR: {lr:.6f} | Batch: {bt:.3f}s | Total: {total:.0f}min | ' \
                         'ETA: {eta:.0f}min | Loss: {loss:.3f} | ' \
                         'Loss(text/kernel): {loss_text:.3f}/{loss_kernel:.3f} ' \
                         '| IoU(text/kernel): {iou_text:.3f}/{iou_kernel:.3f}'.format(
                batch=iter + 1,
                size=len(train_loader),
                lr=optimizer.param_groups[0]['lr'],
                bt=batch_time.avg,
                total=batch_time.avg * iter / 60.0,
                eta=batch_time.avg * (len(train_loader) - iter) / 60.0,
                loss_text=losses_text.avg,
                loss_kernel=losses_kernels.avg,
                loss=losses.avg,
                iou_text=ious_text.avg,
                iou_kernel=ious_kernel.avg,
            )
            print(output_log)
            sys.stdout.flush()
    return train_loss/len(train_loader)


def model_eval(model, test_loader, cfg):
    model.eval()
    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)
    indx = 0
    for data in tqdm(test_loader):
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(
            cfg=cfg
        ))
        # forward
        with torch.no_grad():
            outputs = model(**data)
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[indx]))  # 文件名
        rf.write_result(image_name, outputs)
        indx += 1
    # gt路径
    result_dict = cal_recall_precison_f1('./data/ICDAR2015/test/gt', './outputs/submit_ic15')
    print(result_dict)
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule
    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(model,optimizer, epoch, checkpoint_path, cfg):

    state = dict(
        epoch=epoch + 1,
        iter=0,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
    # file_path = osp.join(checkpoint_path, 'checkpoint.pth.tar')
    torch.save(state, checkpoint_path)

    # if cfg.data.train.type in ['synth'] or \
    #         (state['iter'] == 0 and state['epoch'] > cfg.train_cfg.epoch - 100 and state['epoch'] % 10 == 0):
    #     file_name = 'checkpoint_%dep.pth.tar' % state['epoch']
    #     file_path = osp.join(checkpoint_path, file_name)
    #     torch.save(state, file_path)


def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    print(json.dumps(cfg._cfg_dict, indent=4))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('checkpoints', cfg_name)
    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    print('Checkpoint path: %s.' % checkpoint_path)
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.train)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )
    data_loader_T = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader_T,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )


    # model
    model = build_model(cfg.model)
    model = torch.nn.DataParallel(model).cuda()

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        if cfg.train_cfg.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train_cfg.lr, momentum=0.99, weight_decay=5e-4)
        elif cfg.train_cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_cfg.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=600, eta_min=0.000001)

    start_epoch = 0
    start_iter = 0
    if hasattr(cfg.train_cfg, 'pretrain'):
        assert osp.isfile(cfg.train_cfg.pretrain), 'Error: no pretrained weights found!'
        print('Finetuning from pretrained model %s.' % cfg.train_cfg.pretrain)
        checkpoint = torch.load(cfg.train_cfg.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
    if args.resume:
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        print('Resuming from checkpoint %s.' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    best_model = {'recall': 0, 'precision': 0, 'f1': 0, 'models': ''}
    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))

        train_loss = train(train_loader, model, optimizer, epoch, start_iter, cfg,)
        net_save_path = osp.join(checkpoint_path,'checkpoint.pth.tar')

        if (0.13 < train_loss < 0.15 and epoch % 5 == 0) or train_loss < 0.12:
            recall, precision, f1 = model_eval(model, test_loader, cfg)
            if f1 > best_model['f1']:
                best_path = glob.glob(osp.join(checkpoint_path,'Best_*.pth.tar'))  # 遍历是否有最好的
                if best_path is not None:
                    for b_path in best_path:
                        if os.path.exists(b_path):
                            os.remove(b_path)  # 有的话就删除
                else:
                    pass
                # 参数替换
                best_model['recall'] = recall
                best_model['precision'] = precision
                best_model['f1'] = f1
                best_save_path = osp.join(checkpoint_path,'Best_{}_f1_{:.6f}.pth.tar'.format(epoch+1,f1))
                save_checkpoint(model,optimizer, epoch, best_save_path, cfg)
        #scheduler.step()
        save_checkpoint(model,optimizer, epoch, net_save_path, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()
    main(args)
