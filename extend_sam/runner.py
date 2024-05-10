from datasets import Iterator
from .utils import Average_Meter, Timer, print_and_save_log, mIoUOnline, BinarymIoUOnline, get_numpy_from_tensor, save_model, write_log, \
    check_folder, one_hot_embedding_3d
import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms
import os
import torch.nn as nn

import einops
import numpy as np
from PIL import Image

import tqdm

class BaseRunner():
    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.train_timer = Timer()
        self.eval_timer = Timer()
        try:
            use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            use_gpu = '0'
        self.the_number_of_gpu = len(use_gpu.split(','))
        self.original_size = self.model.img_adapter.sam_img_encoder.img_size
        if self.the_number_of_gpu > 1:
            self.model = nn.DataParallel(self.model)
    
    def load_checkpoint(self, ckp_path: str):
        print(f'loading {ckp_path}')
        ckp = torch.load(ckp_path)
        if (self.the_number_of_gpu > 1):
            self.model.module.load_state_dict(ckp)
        else:
            self.model.load_state_dict(ckp)
        print('loaded')

    def train(self, cfg):
        raise NotImplementedError()
        
    def _eval(self):
        raise NotImplementedError()

class SemRunner(BaseRunner):
    # def __init__(self, **kwargs):
    #     super().__init__(kwargs)

    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        super().__init__(model, optimizer, losses, train_loader, val_loader, scheduler)
        self.exist_status = ['train', 'eval', 'test']

    def train(self, cfg):
        # initial identify
        train_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
        train_iterator = Iterator(self.train_loader)
        best_valid_mIoU = -1
        model_path = "{cfg.model_folder}/{cfg.experiment_name}/model.pth".format(cfg=cfg)
        log_path = "{cfg.log_folder}/{cfg.experiment_name}/log_file.txt".format(cfg=cfg)
        check_folder(model_path)
        check_folder(log_path)
        writer = None
        if cfg.use_tensorboard is True:
            tensorboard_dir = "{cfg.tensorboard_folder}/{cfg.experiment_name}/tensorboard/".format(cfg=cfg)
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(tensorboard_dir)
        # train
        for iteration in range(cfg.max_iter):
            images, labels = train_iterator.get()
            images, labels = images.cuda(), labels.cuda().long()
            masks_pred, iou_pred = self.model(images)
            masks_pred = F.interpolate(masks_pred, self.original_size, mode="bilinear", align_corners=False)

            total_loss = torch.zeros(1).cuda()
            loss_dict = {}
            self._compute_loss(total_loss, loss_dict, masks_pred, labels, cfg)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_dict['total_loss'] = total_loss.item()
            train_meter.add(loss_dict)

            # log
            if (iteration + 1) % cfg.log_iter == 0:
                write_log(iteration=iteration, log_path=log_path, log_data=train_meter.get(clear=True),
                          status=self.exist_status[0],
                          writer=writer, timer=self.train_timer)
            # eval
            if (iteration + 1) % cfg.eval_iter == 0:
                mIoU, _ = self._eval()
                if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                    best_valid_mIoU = mIoU
                    save_model(self.model, model_path, parallel=self.the_number_of_gpu > 1)
                    print_and_save_log("saved model in {model_path}".format(model_path=model_path), path=log_path)
                log_data = {'mIoU': mIoU, 'best_valid_mIoU': best_valid_mIoU}
                write_log(iteration=iteration, log_path=log_path, log_data=log_data, status=self.exist_status[1],
                          writer=writer, timer=self.eval_timer)
        # final process
        save_model(self.model, model_path, is_final=True, parallel=self.the_number_of_gpu > 1)
        if writer is not None:
            writer.close()

    def test(self):
        pass

    def _eval(self):
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric = mIoUOnline(class_names=class_names)
        with torch.no_grad():
            for index, (images, labels) in enumerate(self.val_loader):
                images = images.cuda()
                labels = labels.cuda()
                masks_pred, iou_pred = self.model(images)
                predictions = torch.argmax(masks_pred, dim=1)
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index].squeeze(0))
                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    eval_metric.add(pred_mask, gt_mask)
        self.model.train()
        return eval_metric.get(clear=True)

    def _compute_loss(self, total_loss, loss_dict, mask_pred, labels, cfg):
        """
        Due to the inputs of losses are different, so if you want to add new losses,
        you may need to modify the process in this function
        """
        loss_cfg = cfg.losses
        for index, item in enumerate(self.losses.items()):
            # item -> (key: loss_name, val: loss)
            real_labels = labels
            if loss_cfg[item[0]].label_one_hot:
                class_num = cfg.model.params.class_num
                real_labels = one_hot_embedding_3d(real_labels, class_num=class_num)
            tmp_loss = item[1](mask_pred, real_labels)
            loss_dict[item[0]] = tmp_loss.item()
            total_loss += loss_cfg[item[0]].weight * tmp_loss





class TextRunner(BaseRunner):
    # def __init__(self, **kwargs):
    #     super().__init__(kwargs)

    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        super().__init__(model, optimizer, losses, train_loader, val_loader, scheduler)
        self.exist_status = ['train', 'eval', 'test']

    def train(self, cfg):
        # initial identify
        train_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
        train_iterator = Iterator(self.train_loader)
        best_valid_mIoU = -1
        model_path = "{cfg.model_folder}/{cfg.experiment_name}/model.pth".format(cfg=cfg)
        model_path_current = "{cfg.model_folder}/{cfg.experiment_name}/model_current.pth".format(cfg=cfg)
        log_path = "{cfg.log_folder}/{cfg.experiment_name}/log_file.txt".format(cfg=cfg)
        check_folder(model_path)
        check_folder(log_path)
        writer = None
        if cfg.use_tensorboard is True:
            tensorboard_dir = "{cfg.tensorboard_folder}/{cfg.experiment_name}/tensorboard/".format(cfg=cfg)
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(tensorboard_dir)
        # train
        for iteration in range(cfg.max_iter):
            batch_data = train_iterator.get()
            images, labels, text_array = batch_data[:3]
            # print(text_array)
            images, labels = images.cuda(), labels.cuda().long()
            
            labels_mask_prompt = None
            if (len(batch_data) > 3):
                labels_mask_prompt = batch_data[3]
                labels_mask_prompt = labels_mask_prompt.cuda().float()
                labels_mask_prompt = labels_mask_prompt.unsqueeze(1) # bs h w -> bs 1 h w

            # print(images.shape, labels.shape)
            bs = images.shape[0]
            masks_pred, iou_pred = self.model(images, text_array, labels_mask_prompt)
            masks_pred = F.interpolate(masks_pred, self.original_size, mode="bilinear", align_corners=False)
            masks_pred = masks_pred[:, 0, :, :] # we have only one output
            

            img_vis = einops.rearrange(images, 'bs c h w -> bs h w c')[0].detach().cpu().numpy().astype(np.float32) * 255
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            label_vis = labels[0].detach().cpu().numpy().astype(np.uint8) * 255
            pred_vis = torch.sigmoid(masks_pred[0].detach().cpu()).numpy().astype(np.float32) * 255
            cv2.imwrite(log_path + '.png', img_vis)
            cv2.imwrite(log_path + '_label.png', label_vis)
            cv2.imwrite(log_path + '_pred.png', pred_vis)
            with open(log_path + '_meta.txt', 'w') as f:
                f.write(text_array[0])
            # zx

            total_loss = torch.zeros(1).cuda()
            loss_dict = {}
            self._compute_loss(total_loss, loss_dict, masks_pred, labels, cfg)
            self.optimizer.zero_grad()
            total_loss.backward()

            # # see where the grad goes
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None and param.grad.abs().sum() > 1e-9:
            #         print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")
            # see

            self.optimizer.step()
            self.scheduler.step()
            loss_dict['total_loss'] = total_loss.item()
            train_meter.add(loss_dict)

            # log
            if (iteration + 1) % cfg.log_iter == 0:
                write_log(iteration=iteration, log_path=log_path, log_data=train_meter.get(clear=True),
                          status=self.exist_status[0],
                          writer=writer, timer=self.train_timer)
            # eval
            if (iteration + 1) % cfg.eval_iter == 0:

                # save a model anyway
                save_model(self.model, model_path_current, parallel=self.the_number_of_gpu > 1)
                print_and_save_log("saved model in {model_path}".format(model_path=model_path_current), path=log_path)

                # eval
                mIoU, _ = self._eval()

                if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                    best_valid_mIoU = mIoU
                    save_model(self.model, model_path, parallel=self.the_number_of_gpu > 1)
                    print_and_save_log("saved model in {model_path}".format(model_path=model_path), path=log_path)
                log_data = {'mIoU': mIoU, 'best_valid_mIoU': best_valid_mIoU}
                write_log(iteration=iteration, log_path=log_path, log_data=log_data, status=self.exist_status[1],
                          writer=writer, timer=self.eval_timer)
        # final process
        save_model(self.model, model_path, is_final=True, parallel=self.the_number_of_gpu > 1)
        if writer is not None:
            writer.close()

    def test(self):
        pass

    def _eval(self, dump_dir=None):
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric = BinarymIoUOnline()
        eval_metric_for_each_class = {class_name: BinarymIoUOnline() for class_name in class_names}

        pbar = tqdm.tqdm(self.val_loader)

        i = 0
        
        with torch.no_grad():
            for index, batch_data in enumerate(pbar):
                
                images, labels, text_array = batch_data[:3]
                images = images.cuda() # bs 3 h w
                labels = labels.cuda() # bs h w
                
                labels_mask_prompt = None
                if (len(batch_data) > 3):
                    labels_mask_prompt = batch_data[3]
                    labels_mask_prompt = labels_mask_prompt.cuda().float()
                    labels_mask_prompt = labels_mask_prompt.unsqueeze(1) # bs h w -> bs 1 h w

                # print(images.view(-1)[500000:500020])
                # print(text_array)
                masks_pred, iou_pred = self.model(images, text_array, labels_mask_prompt)
                masks_pred = masks_pred[:, 0, :, :] # we have only one output
                predictions = (masks_pred > 0)
                for batch_index in range(images.shape[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index].squeeze(0))
                    class_name = text_array[batch_index]
                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    # cv2.imwrite('zgt_mask.png', gt_mask * 255)
                    # cv2.imwrite('zpred_mask.png', pred_mask * 255)

                    eval_metric.add(pred_mask, gt_mask)
                    eval_metric_for_each_class[class_name].add(pred_mask, gt_mask)
                
                # i += 1
                # if (i >= 1):
                #     xxx

                if (dump_dir is not None):
                    os.makedirs(dump_dir, exist_ok=True)

                    for batch_index in range(images.shape[0]):
                        
                        img_vis = einops.rearrange(images, 'bs c h w -> bs h w c')[batch_index].detach().cpu().numpy().astype(np.float32) * 255
                        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                        label_vis = labels[batch_index].detach().cpu().numpy().astype(np.uint8) * 255
                        pred_vis = torch.sigmoid(masks_pred[batch_index].detach().cpu()).numpy().astype(np.float32) * 255
                        pred_vis_bin = (torch.sigmoid(masks_pred[batch_index].detach().cpu()).numpy() > 0.5).astype(np.float32) * 255
                        cv2.imwrite(os.path.join(dump_dir, f'{index:04d}_{batch_index}_img_vis.png'), img_vis)
                        cv2.imwrite(os.path.join(dump_dir, f'{index:04d}_{batch_index}_label_vis.png'), label_vis)
                        cv2.imwrite(os.path.join(dump_dir, f'{index:04d}_{batch_index}_pred_vis.png'), pred_vis)
                        cv2.imwrite(os.path.join(dump_dir, f'{index:04d}_{batch_index}_pred_vis_bin.png'), pred_vis_bin)
                        with open(os.path.join(dump_dir, f'{index:04d}_{batch_index}_text.txt'), 'w') as f:
                            f.write(text_array[0])

                    # import numpy as np
                    # print(f'{text_array[batch_index]}, TP: {np.sum(pred_mask * gt_mask)}, pred {np.sum(pred_mask)}, gt {np.sum(gt_mask)}')

        self.model.train()

        total_mIoU, _ = eval_metric.get(clear=True)
        per_class_mIoU = [eval_metric_for_each_class[class_name].get(clear=True)[0] for class_name in class_names]
        return total_mIoU, per_class_mIoU

    def _compute_loss(self, total_loss, loss_dict, mask_pred, labels, cfg):
        """
        labels: 0/1
        mask_pred: float 
        loss: bce: bce with logits
        """
        loss_cfg_dict = cfg.losses
        '''
  losses:
    bce:
      weight: 0.5
    '''
        for index, item in enumerate(self.losses.items()):
            # item -> (key: loss_name, val: loss)
            loss_name, loss = item
            loss_cfg = loss_cfg_dict[loss_name]
            real_labels = labels
            # print(f'loss: {loss_name}, {mask_pred.dtype}, {real_labels.dtype}, {(mask_pred.min(), mask_pred.max())} {(real_labels.min(), real_labels.max())}')
            tmp_loss = loss(mask_pred, real_labels.float())
            loss_dict[loss_name] = tmp_loss.item()
            total_loss += loss_cfg.weight * tmp_loss
        
        # my_loss = torch.nn.BCELoss()(torch.sigmoid(mask_pred), labels.float())
        # true_samples = mask_pred[labels == 1]
        # false_samples = mask_pred[labels == 0]
        # print(f'{len(true_samples)} true, {len(false_samples)} false')
        # print(f'loss={my_loss.item()} vs {total_loss.item()}')

    def run_one_image(self, img_path, prompt: str):

        self.model.eval()

        img = Image.open(img_path).convert('RGB')
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((1024, 1024))
        ])
        img_torch = transform(img).cuda()

        img_torch_bs1 = img_torch.unsqueeze(0)

        # print(img_torch)
        print(img_torch.shape)

        masks_pred, iou_pred = self.model(img_torch_bs1, [prompt])

        masks_pred = F.interpolate(masks_pred, self.original_size, mode="bilinear", align_corners=False)
        masks_pred = masks_pred[:, 0, :, :] # we have only one output
        

        pred_vis = torch.sigmoid(masks_pred[0].detach().cpu()).numpy().astype(np.float32) * 255
        pred_vis_bin = (torch.sigmoid(masks_pred[0].detach().cpu()).numpy() > 0.5).astype(np.float32) * 255
        cv2.imwrite(os.path.join('.', f'one_img_pred_vis.png'), pred_vis)
        cv2.imwrite(os.path.join('.', f'one_img_pred_vis_bin.png'), pred_vis_bin)
        
        




