# !/usr/bin/python
# @File: Losses.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022/5/23 16:17
# @Software: PyCharm
import torch
from torch import nn


class CEContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, base_temperature=0.5, eps=1e-6):
        super(CEContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
        self.base_temperature = base_temperature
        self.smi_fun = nn.CosineSimilarity(dim=-1, eps=eps)

    def forward(self, fea_pro, fea_lg, labels):
        """
        :param fea_pro:[bsz, d_model]
        :param fea_lg:[bsz, d_model]
        :param labels:[batch_size]
        :return:
        """
        bsz = fea_pro.size(0)
        # sim-1 is for numerical stability
        anchor_fea = torch.cat([fea_pro, fea_lg], dim=0)
        contrast_fea = anchor_fea
        anchor_dot_contrast = self.smi_fun(anchor_fea, contrast_fea) / self.temperature
        logits = (anchor_dot_contrast - anchor_dot_contrast.max(dim=-1, keepdim=True).values.detach())
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).to(logits)
        mask = mask.repeat(2, 2)
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = logits_mask * mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos
        return loss.mean()
