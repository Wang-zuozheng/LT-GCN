import copy
import os

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast  # type: ignore
from torchvision import transforms

from config import cfg
from log import logger
from model import SCPNet, load_clip_model
from utils import COCO_missing_val_dataset, CocoDetection, ModelEma, get_ema_co, CustomDataset
from asl import *
from dbl import ResampleLoss
from randaugment import RandAugment


class WeakStrongDataset(torch.utils.data.Dataset):  # type: ignore

    def __init__(self,
                 root,
                 annFile,
                 transform,
                 target_transform=None,
                 class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        self.name = names
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform
        self.strong_transform: transforms.Compose = copy.deepcopy(
            transform)  # type: ignore
        self.strong_transform.transforms.insert(0,
                                                RandAugment(3,
                                                            5))  # type: ignore

    def __getitem__(self, index):
        name = self.name[index]
        
        # path = name.strip('\n').split(',')[0]
        # num = name.strip('\n').split(',')[1]
        # num = num.strip(' ').split(' ')
        
        temp = name.strip('\n').split(' ')
        path = temp[0]
        if 'COCO' in self.root:
            num = temp[1: len(temp)-1]
        elif 'voc' in self.root:
            num = temp[1: ]
        else:
            ValueError
        
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        img_w = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)  # type: ignore # noqa
        assert (self.target_transform is None)
        return [index, img_w,
                self.transform(img),
                self.strong_transform(img)], label

    def __len__(self):
        return len(self.name)


def build_weak_strong_dataset(train_preprocess,
                              val_preprocess,
                              pin_memory=True):
    if "COCO" in cfg.data:
        return build_coco_weak_strong_dataset(train_preprocess, val_preprocess), True
    elif "nuswide" in cfg.data:
        return build_nuswide_weak_strong_dataset(train_preprocess,
                                                 val_preprocess), True
    elif "voc" in cfg.data:
        return build_voc_weak_strong_dataset(train_preprocess, val_preprocess), True
    elif "cub" in cfg.data:
        return build_cub_weak_strong_dataset(train_preprocess, val_preprocess), True
    
    else:
        assert (False)


def build_coco_weak_strong_dataset(train_preprocess, val_preprocess):

    # COCO Data loading
    instances_path_val = os.path.join(cfg.data,
                                      'annotations/instances_val2017.json')
    # instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    instances_path_train = cfg.dataset
    
    # instances_path_train = cfg.train_dataset
    # instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}/val2017'  # args.data
    data_path_train = f'{cfg.data}/train2017'  # args.data
    val_dataset = CocoDetection(data_path_val, instances_path_val,
                                val_preprocess)
    train_dataset = WeakStrongDataset(data_path_train,
                                      instances_path_train,
                                      train_preprocess,
                                      class_num=cfg.num_classes)

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False)

    return [train_loader, val_loader]


def build_nuswide_weak_strong_dataset(train_preprocess, val_preprocess):
    # Nus_wide Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}images'  # args.data
    data_path_train = f'{cfg.data}images'  # args.data

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = WeakStrongDataset(data_path_train,
                                      instances_path_train,
                                      train_preprocess,
                                      class_num=cfg.num_classes)
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    return [train_loader, val_loader]


def build_voc_lt_dataset(train_preprocess, val_preprocess):
    # VOC Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}VOC2012/JPEGImages'  # args.data
    data_path_train = f'{cfg.data}VOC2012/JPEGImages'  # args.data
    dataset = 'voc-lt'
    val_dataset = CustomDataset(
        dataset=dataset, 
        preprocess=val_preprocess,
        split='test'
        )
    train_dataset = CustomDataset(
        dataset=dataset, 
        preprocess=train_preprocess,
        split='train'
        )
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    return [train_loader, val_loader]

def build_voc_weak_strong_dataset(train_preprocess, val_preprocess):
    # VOC Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}VOC2007/JPEGImages'  # args.data
    data_path_train = f'{cfg.data}VOC2012/JPEGImages'  # args.data

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = WeakStrongDataset(data_path_train,
                                      instances_path_train,
                                      train_preprocess,
                                      class_num=cfg.num_classes)
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    return [train_loader, val_loader]

def build_cub_weak_strong_dataset(train_preprocess, val_preprocess):
    # CUB Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}CUB_200_2011/images'  # args.data
    data_path_train = f'{cfg.data}CUB_200_2011/images'  # args.data

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = WeakStrongDataset(data_path_train,
                                      instances_path_train,
                                      train_preprocess,
                                      class_num=cfg.num_classes)
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    return [train_loader, val_loader]

class SCPNetTrainer():

    def __init__(self) -> None:
        super().__init__()

        # bulid dataloader
        # image_size = clip_model.visual.input_resolution
        image_size = cfg.image_size

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                         (0.26862954, 0.26130258, 0.27577711))

        train_preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(), normalize
        ])
        val_preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(), normalize
        ])
        
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
       
        loader, flag = build_weak_strong_dataset(
            train_preprocess,  # type: ignore
            val_preprocess)
        self.train_loader = loader[0]
        self.val_loader = loader[1]

        if flag:
            classnames = self.val_loader.dataset.labels()
        else:
            classnames = []
            for item in loader[1].dataset.categories:
                classnames.append(item['name'])
        assert (len(classnames) == cfg.num_classes)

        # build model
        clip_model, _ = load_clip_model()
        self.model = SCPNet(classnames, clip_model)
        self.relation = self.model.relation
        self.classnames = classnames
        for name, param in self.model.named_parameters():
            if "text_encoder" in name:
                param.requires_grad_(False)
            # if "prompt_learner" not in name:
            #     param.requires_grad = False

        self.model.cuda()
        ema_co = get_ema_co()
        self.ema = ModelEma(self.model, ema_co)  # 0.9997^641=0.82

        self.selected_label = torch.zeros(
            (len(self.train_loader.dataset), cfg.num_classes),
            dtype=torch.long,
        )
        self.selected_label = self.selected_label.cuda()
        self.classwise_acc = torch.zeros((cfg.num_classes, )).cuda()
        self.classwise_acc[:] = 1/cfg.num_classes

    def consistency_loss(self, logits_s, logits_w, y_lb, weight):
        logits_w = logits_w.detach()

        pseudo_label = torch.sigmoid(logits_w)
        pseudo_label_s = torch.sigmoid(logits_s)

        relation_p = pseudo_label @ self.relation.cuda().t()
        pos_mask = relation_p * y_lb
        neg_mask = (1-relation_p) * (1 - y_lb)
        mask_probs = pos_mask + neg_mask
        import mmcv 
        freq_file ='/home/wzz/LMPT/data/coco/class_freq.pkl'
        temp = torch.from_numpy(np.asarray(mmcv.load(freq_file)['class_freq'])).to(torch.float32).cuda()
        weight =  (1 / temp) ** 1 / torch.sum((1 / temp) ** 1)        
        pos_mask_s = pseudo_label_s * y_lb
        neg_mask_s = (1-pseudo_label_s) * (1 - y_lb)
        mask_probs_s = pos_mask_s + neg_mask_s
        # from sklearn.metrics import average_precision_score, roc_auc_score
        # from utils import average_precision
        mask = mask_probs.ge(0.5).float().sum(dim=1) >= mask_probs_s.ge(0.5).float().sum(dim=1)  # convex 样本中有其中一个类大于阈值时，则选中样本
        # a = roc_auc_score(np.array(y_lb.cpu()), np.array(relation_p.cpu()), average=None)
        xs_pos = pseudo_label_s
        xs_neg = 1 - pseudo_label_s
        loss_kl = weight * (relation_p * torch.log(xs_pos.clamp(min=1e-8)) + (1 - relation_p) * torch.log(xs_neg.clamp(min=1e-8))) * mask.reshape(-1, 1)
        return  - cfg.kl_lambda * loss_kl.sum() #-loss.sum()


    def train(self, input, target, criterion, epoch, epoch_i) -> torch.Tensor:
        
        x_ulb_idx, x_lb, x_ulb_w, x_ulb_s = input
        # x_lb, x_ulb_w, x_ulb_s = input
        y_lb = target

        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(), x_ulb_w.cuda(), x_ulb_s.cuda()
        # x_ulb_idx = x_ulb_idx.cuda()
        pseudo_counter = self.selected_label.sum(dim=0)# shape 20
        max_v = pseudo_counter.max().item()
        sum_v = pseudo_counter.sum().item()
        if max_v >= 1:  # not all(5w) -1
            for i in range(cfg.num_classes):
                self.classwise_acc[i] = max(pseudo_counter[i] / max(
                    max_v,
                    cfg.hard_k * len(self.selected_label) - sum_v), 1/cfg.num_classes)

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

        # inference and calculate sup/unsup losses
        with autocast():
            logits, ss_loss = self.model(inputs, 'train', y_lb)
            # logits = self.model(x_lb)
            # logits_s = logits
            # logits_w = self.model(x_ulb_w)
            # # logits_s = self.model(x_ulb_s)
            # logits = torch.cat((logits, logits_w, logits_s))
            
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        logits_x_lb = logits_x_lb.float()
        logits_x_ulb_w, logits_x_ulb_s = logits_x_ulb_w.float(
        ), logits_x_ulb_s.float()
        
        
        
        # sup_loss, _ = criterion(logits_x_lb, y_lb, epoch)
        
        
        
        
        
        # freq_file='/home/wzz/LMPT/data/voc/class_freq.pkl'
        # loss_function = ResampleLoss(
        #         use_sigmoid=True,
        #         reweight_func='rebalance',
        #         focal=dict(focal=True, balance_param=2.0, gamma=2),
        #         logit_reg=dict(neg_scale=5.0, init_bias=0.05),
        #         map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
        #         loss_weight=1.0, freq_file=freq_file
        #     )
        
        freq_file = '/home/wzz/LMPT/data/coco/class_freq.pkl'
        loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=1.0, freq_file=freq_file
            )
        cls_loss, weight_label = loss_function(logits_x_lb, y_lb)
        
        unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                                   logits_x_ulb_w, y_lb, weight_label)
        # loss_asl = ASLloss()(logits_x_lb, y_lb)
        
        # assert (labels is not None)
        # select_mask = labels.sum(dim=1) >= 1
        # if x_ulb_idx[select_mask].nelement() != 0:
        #     self.selected_label[
        #         x_ulb_idx[select_mask]] = labels[select_mask]

        total_loss = cls_loss + unsup_loss
        
        return total_loss #+ cfg.lambda_u * ()
