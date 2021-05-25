import numpy as np

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from config import cfg
from utils import *
import time
from tqdm import tqdm


class Trainer():
    def __init__(self, dataloader, cfg_data, net_fun):

        self.cfg_data = cfg_data

        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH

        self.net_name = cfg.NET
        self.net = net_fun()
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name)

        self.i_tb = 0
        self.epoch = -1

        self.train_loader, self.val_loader = dataloader()

    def train(self):

        # self.validate_V1()
        for epoch in range(cfg.INIT_EPOCH, cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()

            # training
            self.timer['train time'].tic()
            self.forward_dataset()
            self.timer['train time'].toc(average=False)

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)

            # validation
            if epoch % cfg.VAL_FREQ == 0 or epoch > cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                self.validate()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def forward_dataset(self):  # training for all datasets
        self.net.train()
        out_loss = 0
        time = 0
        norm_gt_count = 0
        norm_pred_count = 0

        tk_train = tqdm(
            enumerate(self.train_loader, 0), leave=False, bar_format='{l_bar}{bar:32}{r_bar}',
            colour='#ff0de7', desc='Train Epoch %d/%d' % (self.epoch + 1, cfg.MAX_EPOCH)
        )
        postfix = {'loss': out_loss, 'lr': self.optimizer.param_groups[0]['lr'], 'time': time, 'gt count': norm_gt_count, 'pred count': norm_pred_count}
        # postfix = '[loss: %.4f, lr %.4f, Time: %.2fs, gt count: %.1f pred_count: %.2f]' % \
        #     (out_loss, self.optimizer.param_groups[0]['lr'], time, norm_gt_count, norm_pred_count)
        tk_train.set_postfix(postfix, refresh=True)

        for i, data in tk_train:
            self.timer['iter time'].tic()
            img, gt = data
            img = img.to(cfg.DEVICE)
            gt = gt.to(cfg.DEVICE)

            self.optimizer.zero_grad()
            pred_den = self.net.predict(img)
            loss = self.net.build_loss(pred_den, gt)
            loss.backward()
            self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                """                
                print('[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                      (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr'] * 10000,
                       self.timer['iter time'].diff),
                      '        [cnt: gt: %.1f pred: %.2f]' % (
                          torch.mean(gt.data) / self.cfg_data.LOG_PARA,
                          torch.mean(pred_count.data) / self.cfg_data.LOG_PARA))"""
                out_loss = loss.item()
                time = self.timer['iter time'].diff
                norm_gt_count = torch.mean(torch.sum(gt, dim=(1, 2))).data / self.cfg_data.LOG_PARA
                norm_pred_count = torch.mean(torch.sum(pred_den, dim=(1, 2, 3))).data / self.cfg_data.LOG_PARA
                postfix = {'loss': out_loss, 'lr': self.optimizer.param_groups[0]['lr'], 'time': time,
                           'gt count': norm_gt_count.item(), 'pred count': norm_pred_count.item()}
                tk_train.set_postfix(postfix, refresh=True)

    def validate(self):

        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        time_sampe = 0
        step = 0

        tk_valid = tqdm(
            enumerate(self.val_loader, 0), leave=False, bar_format='{l_bar}{bar:32}{r_bar}',
            desc='Validating'
        )

        for vi, data in tk_valid:
            img, gt = data

            with torch.no_grad():
                img = img.to(cfg.DEVICE)
                gt = gt.to(cfg.DEVICE)

                step = step + 1
                time_start1 = time.time()
                pred_map = self.net.predict(img)
                time_end1 = time.time()
                self.net.build_loss(pred_map, gt)
                time_sampe = time_sampe + (time_end1 - time_start1)

                pred_map = pred_map.data.cpu().numpy()
                gt = gt.data.cpu().numpy()

                pred_cnt = torch.sum(pred_map) / self.cfg_data.LOG_PARA
                gt_count = torch.sum(gt) / self.cfg_data.LOG_PARA

                losses.update(self.net.loss.item())
                maes.update(abs(gt_count - pred_cnt))
                mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net, self.epoch, self.exp_path, self.exp_name, [mae, mse, loss],
                                         self.train_record, self.log_txt)
        print_summary(self.exp_name, [mae, mse, loss], self.train_record)
        print('\nForward Time: %fms' % (time_sampe * 1000 / step))
