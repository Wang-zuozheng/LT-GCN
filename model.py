from collections import OrderedDict
import mmcv
import torch
import torch.nn as nn
from torch.nn import functional as F
from clip.model import build_model_conv_proj
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from config import cfg
from log import logger

_tokenizer = _Tokenizer()

def load_clip_to_cpu():
    backbone_name = 'RN50'
    # backbone_name = 'ViT-B/16'
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(  # type: ignore
            model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())  # type: ignore
    # model = build_model_conv_proj(state_dict or model.state_dict(), cfg)

    return model


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.n_ctx
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        ctx_init = cfg.ctx_init.replace("_", " ")
        assert (n_ctx == len(ctx_init.split(" ")))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        prompt_prefix = ctx_init

        self.ctx = nn.Parameter(ctx_vectors)  # type: ignore
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_middle", embedding[:, 1:(1 + n_ctx), :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],  # type: ignore
            dim=1,
        )
        return prompts

def load_clip_model():
    clip_model = load_clip_to_cpu()

    # CLIP's default precision is fp16
    clip_model.float()
    return clip_model, clip._transform(clip_model.visual.input_resolution)

import math
import numpy as np
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# from timm.models.vision_transformer import resize_pos_embed
class SCPNet(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_class = cfg.num_classes 
        self.q_fc = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024))
        self.k_fc = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024))
        self.pro = torch.zeros(self.num_class, 1024)
        
        self.gc1 = GraphConvolution(1024, 2048)
        self.gc2 = GraphConvolution(2048, 2048)
        self.gc3 = GraphConvolution(2048, 1024)
        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)
        
        # self.q_fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        # self.k_fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        # self.pro = torch.zeros(self.num_class, 512)
        
        # self.gc1 = GraphConvolution(512, 1024)
        # self.gc2 = GraphConvolution(1024, 1024)
        # self.gc3 = GraphConvolution(1024, 512)
        # self.relu = nn.LeakyReLU(0.2)
        # self.relu2 = nn.LeakyReLU(0.2)
        
        # self.relation = torch.Tensor(np.load('data_x.npy'))
        self.relation = torch.Tensor(np.load(cfg.relation_file))
        
        _ ,max_idx = torch.topk(self.relation, cfg.sparse_topk)
        mask = torch.ones_like(self.relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        self.relation[mask] = 0
        sparse_mask = mask
        dialog = torch.eye(cfg.num_classes).type(torch.bool)
        self.relation[dialog] = 0
        self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1) * cfg.reweight_p
        self.relation[dialog] = 1-cfg.reweight_p

        self.gcn_relation = self.relation.clone()
        assert(self.gcn_relation.requires_grad == False)
        self.relation = torch.exp(self.relation/cfg.T) / torch.sum(torch.exp(self.relation/cfg.T), dim=1).reshape(-1,1)
        self.relation[sparse_mask] = 0
        self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1)

    def forward(self, image, type='train', gt_labels=None):
        tokenized_prompts = self.tokenized_prompts
        image_features = self.image_encoder(image.type(self.dtype))
        
        if type == 'train':
            num_lb = image_features.shape[0] // 3
            image_features_lb = image_features[:num_lb]
            image_features_ulb_w, image_features_ulb_s = image_features[num_lb:].chunk(2)
            
            
            label_pro = torch.eye(self.num_class)
            
            
            pro = self.k_fc(image_features_lb) 
            pro = torch.mm(torch.mm(torch.linalg.pinv(torch.mm(gt_labels.T.float(), gt_labels.float()).float()), gt_labels.T.float()), pro.float()).cpu()
            self.pro = 0.99 * self.pro + (1 - 0.99) * pro
            pro = nn.functional.normalize(self.pro, dim=1)
            
            q = self.q_fc(image_features_ulb_s)
            q = nn.functional.normalize(q, dim=1)
            
            k = self.q_fc(image_features_ulb_w)
            k = nn.functional.normalize(k, dim=1)
            
            image_features = image_features / image_features.norm(dim=-1,
                                                                keepdim=True)
            
            features = torch.cat((q, k, pro.cuda().detach()), dim=0)#, self.queue.clone().detach().t()
            labels = torch.cat((gt_labels, gt_labels, label_pro.cuda().detach()), dim=0)# 576,20, self.label_queue.clone().detach()
            batch_size = gt_labels.shape[0]
            longth = 2*batch_size + self.num_class    
            
            # 无监督学习的mask
            mask = torch.zeros((batch_size, longth)).cuda()
            for i in range(batch_size):
                mask[i, i+batch_size] = 1
                
            # label之间的相似度
            and_min = torch.matmul(gt_labels.int().cpu(),labels.T.int().cpu()) # 与操作
            or_max = (gt_labels.sum(1).repeat(longth,1).T + labels.sum(1)).cpu()-and_min  # 或操作 32,576 + 576,1 
            sim =  and_min/or_max # 完全一致为1
            
            thr = 0 #0.5
            mask_pos = torch.where(sim >= thr , sim, 0).cuda()
            
            mask_neg = torch.where(mask_pos==1, mask_pos,  1 - mask_pos)
            # 不同类别设置不同T
            class_split = mmcv.load('/home/wzz/new/DistributionBalancedLoss/appendix/VOCdevkit/longtail2012/class_split.pkl')
            # class_split = mmcv.load('/home/wzz/new/DistributionBalancedLoss/appendix/coco/longtail2017/class_split.pkl')
            M_index = gt_labels[:,list(class_split['middle'])].sum(1)
            M_index = torch.sign(M_index) 
            mask_m = torch.unsqueeze(M_index, dim=1)
            mask_m = mask_m.repeat(1,longth)
            T_index = gt_labels[:,list(class_split['tail'])].sum(1)
            T_index = torch.sign(T_index) 
            mask_tail = torch.unsqueeze(T_index, dim=1) 
            
            mask_tail = mask_tail.repeat(1,longth) 
            mask_m -= mask_tail     
            mask_m = torch.where(mask_m<0,0,mask_m)
            # compute logits
            anchor_dot_contrast_head = (1-mask_tail-mask_m) * torch.div(
                torch.matmul(features[:batch_size], features.T),
                0.7)#1.0
            anchor_dot_contrast_middle = (mask_m) * torch.div(
                torch.matmul(features[:batch_size], features.T),
                0.7)#1.0
            anchor_dot_contrast_tail = mask_tail * torch.div(
                torch.matmul(features[:batch_size], features.T),
                0.7)# 0.3
            anchor_dot_contrast = anchor_dot_contrast_head +anchor_dot_contrast_tail+anchor_dot_contrast_middle
            # compute logits负
            logits = anchor_dot_contrast
            
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            logits = logits - logits_max.detach()
            
            # compute logits正
            anchor_dot_contrast_pos = torch.div(
                torch.matmul(features[:batch_size], features.T),
                0.7)#0.5
            logits_pos = anchor_dot_contrast_pos
            logits_max, _ = torch.max(logits_pos, dim=1, keepdim=True)
            logits_pos = logits_pos - logits_max.detach()
            
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).cuda(),
                0
            ).cuda()

            mask = mask * logits_mask# 无监督的mask
            
            mask_pos = mask_pos * logits_mask# 有监督的 多标记的mask
            miu = 1
            mask = torch.where(mask==1, mask, miu * mask_pos)
            
            per_ins_weight = labels.sum(0)
            
            balanced_neg_weight = []
            for i in range(batch_size):
                per_ins_weight_i = (mask_neg[i].unsqueeze(1).repeat(1,self.num_class) * labels).sum(0)
                sample_per_balanced_neg_weight = list((labels / (per_ins_weight_i)).sum(1)) 
                balanced_neg_weight.append(sample_per_balanced_neg_weight)
            # 对比损失 
            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask  * mask_neg
            # exp_logits = torch.exp(logits* ((labels / (per_ins_weight+1)).sum(1))) * logits_mask
            # exp_logits = exp_logits * ((labels / (per_ins_weight+1)).sum(1))
            exp_logits = exp_logits * torch.tensor(balanced_neg_weight).cuda()
            log_prob = logits_pos - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12) # 32, 576 - 32, 1

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = - (mask * log_prob).sum(1) / mask.sum(1)
            
            
            ss_loss = mean_log_prob_pos
        
        logit_scale = self.logit_scale.exp()
        if cfg.scale != 'clip':
            assert(isinstance(cfg.scale, int))
            logit_scale = cfg.scale
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        identity = text_features

        text_features = self.gc1(text_features, self.gcn_relation.cuda())
        text_features = self.relu(text_features)
        text_features = self.gc2(text_features, self.gcn_relation.cuda())
        text_features = self.relu2(text_features)
        text_features = self.gc3(text_features, self.gcn_relation.cuda())
        b = 1
        text_features = b * text_features + identity
        
        # text_features = identity
        
        # class_split = mmcv.load('/home/wzz/new/DistributionBalancedLoss/appendix/VOCdevkit/longtail2012/class_split.pkl')
        # # class_split = mmcv.load('/home/wzz/new/DistributionBalancedLoss/appendix/coco/longtail2017/class_split.pkl')
        # mask_tail_feature = torch.zeros(text_features.shape[0], text_features.shape[1])
        # for i in class_split['tail']:
        #     mask_tail_feature[i, :] = text_features[i, :]
        # text_features = identity + mask_tail_feature.cuda()
        
        image_features= image_features / image_features.norm(dim=-1,
                                                                keepdim=True)    
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        
        # # Class-Specific Region Feature Aggregation
        # output = F.conv1d(image_features, text_features[:, :, None])#8 80 50
      
        # # WTA
        # # wi = F.softmax(1 * output * torch.max(output, dim=1)[0].unsqueeze(1))
        # # output = torch.mul(wi, output)
        # w = F.softmax(output, dim=-1)# 8 20 50 softmax(Si,m+) 
       
        # logits =  logit_scale * (w * output).sum(-1)#output是 Si,m+ Si,m- 是局部Fi 和Ft的余弦相似度

        if type == 'train':
            return logits, ss_loss.sum()
        else:
            return logits
