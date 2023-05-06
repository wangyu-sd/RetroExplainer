# !/usr/bin/python3
# @File: RetroAGT.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.20.22
import sys

sys.path.append("..")
sys.path.append("./model")
import os
from typing import List
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import pandas as pd
from torch import nn, Tensor
from model.LearningRate import PolynomialDecayLR
from model.Embeddings import RetroAGTEmbeddingLayer
from model.Modules import RetroAGTEncoderLayer, RetroAGTEncoder, MultiHeadAtomAdj
from model.LeavingGroupList import LeavingGroupList
from model import util
# from model.Losses import CEContrastiveLoss
from itertools import chain
from torch.utils import data
from rdkit import Chem
import heapq
import copy
import re
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
from data.datasets import MultiStepDataset
from torch.utils.data import DataLoader


class RetroAGT(pl.LightningModule):
    def __init__(self, d_model=512, nhead=32, num_shared_layer=8, num_rc_layer=4, num_lg_layer=4, num_h_layer=4,
                 num_ct_layer=2, n_rxn_type=10, n_rxn_cnt=7, dim_feedforward=512, dropout=0.1, max_paths=200,
                 n_graph_type=6,
                 max_single_hop=4, max_ct_atom=4, use_contrastive=True, use_adaptive_multi_task=True,
                 atom_dim=90, total_degree=50, formal_charge=30, hybrid=10, exp_valance=20, hydrogen=20, aromatic=2,
                 ring=10, n_layers=1, batch_first=True, known_rxn_type=True, known_rxn_cnt=True, norm_first=False,
                 activation='gelu', warmup_updates=6e4, tot_updates=1e6, peak_lr=2e-4, end_lr=1e-9, weight_decay=0.99,
                 leaving_group_path=None, use_3d_info=False, use_dist_adj=True, dataset_path=None, batch_considered=200):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_shared_layer = num_shared_layer
        self.num_rc_layer = num_rc_layer
        self.num_lg_layer = num_lg_layer
        self.num_h_layer = num_h_layer
        self.num_ct_layer = num_ct_layer
        self.n_rxn_type = n_rxn_type
        self.n_rxn_cnt = n_rxn_cnt
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_paths = max_paths
        self.n_graph_type = n_graph_type
        self.max_single_hop = max_single_hop
        self.max_ct_atom = max_ct_atom
        self.use_contrastive = use_contrastive
        self.use_adaptive_multi_task = use_adaptive_multi_task
        self.atom_dim = atom_dim
        self.total_degree = total_degree
        self.formal_charge = formal_charge
        self.hybrid = hybrid
        self.exp_valance = exp_valance
        self.hydrogen = hydrogen
        self.aromatic = aromatic
        self.ring = ring
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.known_rxn_type = known_rxn_type
        self.known_rxn_cnt = known_rxn_cnt
        self.norm_first = norm_first
        self.activation = activation
        self.warmup_updates = warmup_updates
        self.peak_lr = peak_lr
        self.tot_updates = tot_updates
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.use_3d_info = use_3d_info
        self.n_graph_type = n_graph_type
        self.use_dist_adj = use_dist_adj
        self.max_single_hop = max_single_hop
        self.leaving_group_path = leaving_group_path
        self.dataset_path = dataset_path
        self.dataset = None
        assert os.path.isfile(leaving_group_path)
        self.lg = LeavingGroupList(torch.load(leaving_group_path), n_rxn_type, n_rxn_cnt)
        self.lg_size = len(self.lg)
        self.batch_size = None
        self.batch_considered = batch_considered

        encoder_layer = RetroAGTEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                             batch_first=batch_first, norm_first=norm_first)
        if self.num_shared_layer:
            self.emb = RetroAGTEmbeddingLayer(d_model, nhead, max_paths, n_graph_type, max_single_hop, atom_dim,
                                              total_degree, formal_charge, hybrid, exp_valance, hydrogen, aromatic,
                                              ring, n_layers, need_graph_token=True, use_3d_info=use_3d_info,
                                              dropout=dropout, known_rxn_type=known_rxn_type, n_rxn_cnt=n_rxn_cnt,
                                              known_rxn_cnt=known_rxn_cnt, use_dist_adj=use_dist_adj)
        else:
            self.emb = nn.ModuleList([
                RetroAGTEmbeddingLayer(d_model, nhead, max_paths, n_graph_type, max_single_hop, atom_dim,
                                       total_degree, formal_charge, hybrid, exp_valance, hydrogen, aromatic,
                                       ring, n_layers, need_graph_token=True, use_3d_info=use_3d_info,
                                       dropout=dropout, known_rxn_type=known_rxn_type, n_rxn_cnt=n_rxn_cnt,
                                       known_rxn_cnt=known_rxn_cnt, use_dist_adj=use_dist_adj)
                for _ in range(3)
            ])
            self.rc_encoder = RetroAGTEncoder(encoder_layer, num_rc_layer, nn.LayerNorm(d_model))

        self.shared_encoder = RetroAGTEncoder(encoder_layer, num_shared_layer, nn.LayerNorm(d_model))

        self.rc_adj_fn = MultiHeadAtomAdj(d_model, nhead, batch_first=batch_first)
        self.rc_out_fn = nn.Sequential(
            nn.Linear(nhead, 1)
        )

        self.h_encoder = RetroAGTEncoder(encoder_layer, num_h_layer, nn.LayerNorm(d_model))
        self.h_out_fn = nn.Sequential(
            nn.Linear(d_model, 7)
        )
        # self.rc_activation = nn.Sigmoid()

        # self.rc_ct_fn = nn.Sequential(
        #     nn.Sigmoid(),
        #     nn.Linear(1, 1),
        # )

        self.lg_encoder = RetroAGTEncoder(encoder_layer, num_lg_layer, nn.LayerNorm(d_model))
        # print(num_shared_layer, num_lg_layer)
        self.lg_out_fn = nn.Sequential(
            nn.Linear(d_model, self.lg_size)
        )

        # encoder_layer = RetroAGTEncoderLayer(d_model * 2, nhead, dim_feedforward * 2, dropout, activation,
        #                                      batch_first=batch_first, norm_first=norm_first)
        self.ct_encoder = RetroAGTEncoder(encoder_layer, num_ct_layer, nn.LayerNorm(d_model))
        self.ct_adj_fn = MultiHeadAtomAdj(d_model, nhead, batch_first=batch_first)
        self.ct_out_fn = nn.Sequential(
            nn.Linear(nhead, 1)
        )

        self.gate_embedding = nn.Embedding(10, d_model, padding_idx=0)

        self.criterion_rc = nn.BCEWithLogitsLoss()
        self.criterion_h = nn.CrossEntropyLoss()
        self.criterion_lg = nn.CrossEntropyLoss()
        # self.criterion_con = CEContrastiveLoss()
        self.criterion_ct = nn.BCEWithLogitsLoss()

        self.state = None

        loss_init = torch.zeros(4, batch_considered)
        loss_last = torch.zeros(4, batch_considered // 10)
        loss_last2 = torch.zeros(4, batch_considered // 10)
        cur_loss_step = torch.zeros(1, dtype=torch.long)
        self.register_buffer('loss_init', loss_init)
        self.register_buffer('loss_last', loss_last)
        self.register_buffer('loss_last2', loss_last2)
        self.register_buffer('cur_loss_step', cur_loss_step)

        self.max_lg_consider = 10
        self.max_lg_node = 40
        self.max_gate_num = 4

        self.bond_decoder = {1: Chem.BondType.SINGLE,
                             1.5: Chem.BondType.AROMATIC,
                             2: Chem.BondType.DOUBLE,
                             3: Chem.BondType.TRIPLE
                             }
        self.save_hyperparameters()

    def forward(self, batch, max_lg_node=None):
        """{
            atom_fea:[bsz, n_type, n_atom]
            bond_adj: [bsz, n_atom, n_atom]
            dist_adj:
            center_cnt:
            rxn_type:
            dist3d_adj:
            lg_dic:
        }
        :return:
        """
        pro_dic = batch['product']
        dist_adj_3d = pro_dic['dist_adj_3d'] if self.use_3d_info else None
        atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, = pro_dic['atom_fea'], pro_dic['bond_adj'], \
                                                              pro_dic['dist_adj'], batch['center_cnt'], \
                                                              batch['rxn_type']
        # print('atom_fea:', atom_fea.min(), atom_fea.max())
        # print('bond_adj:', bond_adj.min(), bond_adj.max())
        # print('dist_adj:', dist_adj.min(), dist_adj.max())
        # print('center_cnt:', center_cnt.min(), center_cnt.max())
        # print('rxn_type:', rxn_type.min(), rxn_type.max())

        if not self.known_rxn_cnt:
            center_cnt = torch.zeros_like(center_cnt)
        bsz, n_atom, _ = bond_adj.size()

        if self.num_shared_layer > 0:
            shared_atom_fea, masked_adj = self.emb(atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist_adj_3d)
            shared_atom_fea = self.shared_encoder(shared_atom_fea, masked_adj)
            # rc_fea = self.rc_encoder(shared_atom_fea, masked_adj)
            rc_fea = shared_atom_fea
        else:
            shared_atom_fea, masked_adj = self.emb[0](atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist_adj_3d)
            rc_fea = self.rc_encoder(shared_atom_fea, masked_adj)

        rc_prob = self.rc_adj_fn(rc_fea[:, 1:], rc_fea[:, 1:], None)
        rc_adj_prob = self.rc_out_fn(rc_prob).squeeze() + torch.where(bond_adj > 1, 0., -10.)

        if self.num_shared_layer == 0:
            shared_atom_fea, masked_adj = self.emb[1](atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist_adj_3d)
        h_fea = self.h_encoder(shared_atom_fea, masked_adj)
        h_prob = self.h_out_fn(h_fea[:, 1:])
        h_mask = torch.where(atom_fea[:, 0] > 0, 0., -1e3)
        h_prob[:, :, :3] += h_mask[:, :, None]
        h_prob[:, :, 4:] += h_mask[:, :, None]

        if self.num_shared_layer == 0:
            shared_atom_fea, masked_adj = self.emb[2](atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist_adj_3d)
        lg_fea = self.lg_encoder(shared_atom_fea, masked_adj)[:, 0]
        lg_prob = self.lg_out_fn(lg_fea)

        if self.state == "inference":
            bsz, na_fea_type, _ = pro_dic['atom_fea'].size()
            lg_atom_fea, lg_bond_adj, lg_dist_adj = self.construct_lg(lg_prob, bsz, na_fea_type, max_lg_node)

            shared_atom_fea_lg2, masked_adj_lg2 = self.emb(lg_atom_fea, lg_bond_adj, lg_dist_adj,
                                                           center_cnt, rxn_type, dist3d_adj=None, contrast=True)

            shared_atom_fea_lg2[:, 1:] += self.gate_embedding(batch['gate_token'])
            ct_fea_lg2 = self.ct_encoder(shared_atom_fea_lg2, masked_adj_lg2)[:, 1:]
            # rc_fea = self.ct_encoder(shared_atom_fea, masked_adj)[:, 1:]
            ct_prob2 = self.ct_adj_fn(rc_fea[:, 1:], ct_fea_lg2)[:, :, :self.max_ct_atom]
            ct_prob2 = self.ct_out_fn(ct_prob2).squeeze() + torch.where(atom_fea[:, 0] > 0, 0., -1e3)[:, :, None]

            return {
                "rc_adj_prob": rc_adj_prob,
                "h_prob": h_prob,
                "rxn_type": batch['rxn_type'],
                "lg_prob": lg_prob,
                "ct_prob_pred": ct_prob2,
            }

        lg_dic = batch['lg']


        if self.num_shared_layer > 0:
            shared_atom_fea_lg, masked_adj_lg = self.emb(lg_dic['atom_fea'], lg_dic['bond_adj'], lg_dic['dist_adj'],
                                                         center_cnt, rxn_type, dist3d_adj=None, contrast=True)
            shared_atom_fea_lg = self.shared_encoder(shared_atom_fea_lg, masked_adj_lg)
        else:
            shared_atom_fea_lg, masked_adj_lg = self.emb[2](lg_dic['atom_fea'], lg_dic['bond_adj'], lg_dic['dist_adj'],
                                                            center_cnt, rxn_type, dist3d_adj=None, contrast=True)

        shared_atom_fea_lg[:, 1:] += self.gate_embedding(batch['gate_token'])

        if self.use_contrastive:
            lg_fea2 = self.lg_encoder(shared_atom_fea_lg, masked_adj_lg)[:, 0]
            lg_prob2 = self.lg_out_fn(lg_fea2)
        else:
            lg_prob2 = None
            lg_fea2 = None

        ct_fea_lg = self.ct_encoder(shared_atom_fea_lg, masked_adj_lg)[:, 1:]
        # rc_fea = self.ct_encoder(shared_atom_fea, masked_adj)[:, 1:]
        ct_prob_fea = self.ct_adj_fn(rc_fea[:, 1:], ct_fea_lg, None)[:, :, :self.max_ct_atom]
        ct_prob = self.ct_out_fn(ct_prob_fea).squeeze() + torch.where(atom_fea[:, 0] > 0, 0., -1e3)[:, :, None]
        # ct_prob += torch.bmm(self.rc_ct_fn(rc_adj_prob), ct_prob)
        if self.state == 'predict':
            bond_types = util.bond_fea2type(batch['product']['bond_adj'])
            bsz, _, n_lg_atoms = lg_dic['atom_fea'].size()
            lg_atom_fea, lg_bond_adj, lg_dist_adj = torch.zeros_like(lg_dic['atom_fea']), \
                                                    torch.zeros_like(lg_dic['bond_adj']), \
                                                    torch.zeros_like(lg_dic['dist_adj'])
            gate_token = torch.zeros_like(batch['gate_token'])
            lg_id = self.greedy_search(lg_prob).detach().tolist()
            for idx in range(bsz):
                cur_lg = self.lg[lg_id[idx]]
                cur_na = int(cur_lg.na)
                lg_atom_fea[idx, :, :cur_na] = cur_lg.atom_fea
                lg_bond_adj[idx, :cur_na, :cur_na] = cur_lg.bond_adj
                lg_dist_adj[idx, :cur_na, :cur_na] = cur_lg.dist_adj
                for gi, gn in enumerate(cur_lg.gate_num):
                    gate_token[idx, gi] = gn

            shared_atom_fea_lg2, masked_adj_lg2 = self.emb(lg_atom_fea, lg_bond_adj, lg_dist_adj,
                                                           center_cnt, rxn_type, dist3d_adj=None, contrast=True)

            shared_atom_fea_lg2[:, 1:] += self.gate_embedding(gate_token)
            ct_fea_lg2 = self.ct_encoder(shared_atom_fea_lg2, masked_adj_lg2)[:, 1:]
            # rc_fea = self.ct_encoder(shared_atom_fea, masked_adj)[:, 1:]
            ct_prob2 = self.ct_adj_fn(rc_fea[:, 1:], ct_fea_lg2)[:, :, :self.max_ct_atom]
            ct_prob2 = self.ct_out_fn(ct_prob2).squeeze() + torch.where(atom_fea[:, 0] > 0, 0., -1e3)[:, :, None]

            return {
                "mol_fea": shared_atom_fea[:, 0],
                "masked_adj": masked_adj,
                "atoms": batch['product']['atom_fea'][:, 0],
                "bonds": bond_types,
                "h_prob": h_prob,
                "h_fea": h_fea,
                "rxn_type": batch['rxn_type'],
                "lg_id": batch['lg_id'],
                # "lg_id_pred": self.greedy_search(lg_prob),
                "rc_adj_prob": rc_adj_prob,
                "rc_prob": rc_prob,
                "lg_fea": lg_fea,
                "lg_fea2": lg_fea2,
                "lg_prob": lg_prob,
                # "lg_prob2": lg_prob2,
                "ct_prob_fea": ct_prob_fea,
                "ct_prob": ct_prob,
                "ct_prob_pred": ct_prob2,
            }

        return rc_adj_prob, lg_prob, lg_prob2, ct_prob, h_prob

    def calc_loss(self, batch, rc_adj_prob, lg_prob, lg_prob2, ct_prob, h_prob):
        # print("batch['rc_target']", batch["rc_target"].max(), batch["rc_target"].min())
        # print("batch['lg_id']", batch["lg_id"].max(), batch["lg_id"].min())
        # print("batch['ct_target']", batch["ct_target"].max(), batch["ct_target"].min())
        # print("batch['rc_h']", batch["rc_h"].max(), batch["rc_h"].min())
        bsz, n_atom, _ = rc_adj_prob.size()
        loss_rc = self.criterion_rc(rc_adj_prob.reshape(bsz * n_atom * n_atom),
                                    batch["rc_target"].reshape(bsz * n_atom * n_atom))
        loss_voc = self.criterion_lg(lg_prob, batch['lg_id'])

        if self.use_contrastive:
            loss_voc += self.criterion_lg(lg_prob2, batch['lg_id'])
            loss_voc /= 2
        loss_ct = self.criterion_ct(ct_prob, batch['ct_target'])
        loss_h = self.criterion_h(h_prob.reshape(bsz*n_atom, -1), batch['rc_h'].reshape(bsz*n_atom).long())
        return loss_rc, loss_voc, loss_ct, loss_h

    def calc_mt_loss(self, loss_list):
        loss_list = torch.stack(loss_list)
        if not self.use_adaptive_multi_task or self.num_shared_layer == 0:
            return loss_list.sum()

        if self.cur_loss_step == 0:
            if self.training:
                self.loss_init[:, 0] = loss_list.detach()
                self.loss_last2[:, 0] = loss_list.detach()
                self.cur_loss_step += 1
                loss_t = (loss_list / self.loss_init[:, 0]).mean()
            else:
                loss_t = (loss_list / loss_list.detach()).mean()

        elif self.cur_loss_step == 1:
            if self.training:
                self.loss_last[:, 0] = loss_list.detach()
                self.loss_init[:, 1] = loss_list.detach()
                self.cur_loss_step += 1
                loss_t = (loss_list / self.loss_init[:, :2].mean(dim=-1)).mean()
            else:
                loss_t = (loss_list / loss_list.detach()).mean()
        else:
            cur_loss_init = self.loss_init[:, :self.cur_loss_step].mean(dim=-1)
            cur_loss_last = self.loss_last[:, :self.cur_loss_step - 1].mean(dim=-1)
            cur_loss_last2 = self.loss_last2[:, :self.cur_loss_step - 1].mean(dim=-1)
            w = F.softmax(cur_loss_last / cur_loss_last2, dim=-1).detach()
            loss_t = (loss_list / cur_loss_init * w).sum()

            if self.training:
                cur_init_idx = self.cur_loss_step.item() % self.batch_considered
                self.loss_init[:, cur_init_idx] = loss_list.detach()

                cur_loss_last2_step = (self.cur_loss_step.item() - 1) % (self.batch_considered // 10)
                self.loss_last2[:, cur_loss_last2_step] = self.loss_last[:, cur_loss_last2_step - 1]
                self.loss_last[:, cur_loss_last2_step] = loss_list.detach()
                self.cur_loss_step += 1
        return loss_t

    def training_step(self, batch, batch_idx):
        self.state = 'train'
        rc_adj_prob, lg_prob, lg_prob2, ct_prob, h_prob = self.forward(batch)
        loss_rc, loss_voc, loss_ct, loss_h = self.calc_loss(batch, rc_adj_prob, lg_prob, lg_prob2, ct_prob, h_prob)
        loss_t = self.calc_mt_loss([loss_rc, loss_voc, loss_ct, loss_h])

        if batch_idx == 0:
            all_acc, rc_acc, lg_acc, ct_acc, h_acc = \
                self.calc_batch_accuracy(batch, rc_adj_prob, lg_prob, ct_prob, h_prob)
            self.log("acc_all", all_acc, prog_bar=True, logger=True)
            self.log("acc_rc_h", h_acc, prog_bar=True, logger=True)
            self.log("acc_rc", rc_acc, prog_bar=True, logger=True)
            # self.log("acc_h_atom", rc_h_atom_acc, prog_bar=True, logger=True)
            self.log("acc_ct", ct_acc, prog_bar=False, logger=True)
            self.log("acc_lg", lg_acc, prog_bar=True, logger=True)

        self.log("loss_rc", loss_rc, prog_bar=False, logger=True)
        self.log("loss_voc", loss_voc, prog_bar=False, logger=True)
        # self.log("loss_con", loss_con, prog_bar=False, logger=True)
        # self.log("loss_rc_h", loss_rc_h, prog_bar=False, logger=True)
        self.log("loss_h", loss_h, prog_bar=False, logger=True)
        self.log("loss_ct", loss_ct, prog_bar=False, logger=True)
        self.log("loss_t", loss_t, prog_bar=True, logger=True)

        self.save_hyperparameters()
        return loss_t

    def validation_step(self, batch, batch_idx):
        self.state = 'valid'
        rc_adj_prob, lg_prob, lg_prob2, ct_prob, h_prob = self.forward(batch)
        loss_rc, loss_voc, loss_ct, loss_h = self.calc_loss(batch, rc_adj_prob, lg_prob, lg_prob2, ct_prob, h_prob)
        loss_t = self.calc_mt_loss([loss_rc, loss_voc, loss_ct, loss_h])

        all_acc, rc_acc, lg_acc, ct_acc, h_acc = \
            self.calc_batch_accuracy(batch, rc_adj_prob, lg_prob, ct_prob, h_prob)

        return {
            "valid_loss_rc": loss_rc,
            "valid_loss_voc": loss_voc,
            "valid_loss_ct": loss_ct,
            "valid_loss_h": loss_h,
            "valid_loss": loss_t,
            # "valid_loss_con": loss_con,
            # "valid_acc_h_atoms": rc_h_atom_acc,
            "valid_acc_h": h_acc,
            "valid_acc_ct": ct_acc,
            "valid_acc_all": all_acc,
            "valid_acc_rc": rc_acc,
            "valid_acc_lg": lg_acc,
        }

    def validation_epoch_end(self, outputs):
        self._log_dict(self._avg_dicts(outputs))

    def test_step(self, batch, batch_idx):
        self.state = 'test'
        rc_adj_prob, lg_prob, lg_prob2, ct_prob, h_prob = self.forward(batch)
        all_acc, rc_acc, lg_acc, ct_acc, h_acc, rxn_type_acc = \
            self.calc_beam_search_accuracy(batch, rc_adj_prob, lg_prob, ct_prob, h_prob)
        test_output = {}

        for k in [1, 3, 5, 10]:
            test_output[f"all_top{k}_beam_acc"] = all_acc[f"all_top{k}_acc"]
            # test_output[f"lg_top{k}_beam_acc"] = lg_acc[f"lg_top{k}_acc"]
            # test_output[f"rc_top{k}_beam_acc"] = rc_acc[f"rc_top{k}_acc"]
            # test_output[f"all_top{k}_acc"] = 0

        if rxn_type_acc is not None:
            for i in range(10):
                for k in [1, 3, 5, 10]:
                    if rxn_type_acc[f"rxn_type{i}_top{k}_acc"] is not None:
                        test_output[f"rxn_type{i}_top{k}_beam_acc"] = rxn_type_acc[f"rxn_type{i}_top{k}_acc"]
        test_output['ct_acc'] = ct_acc
        test_output['h_acc'] = h_acc
        
        return test_output

    def test_epoch_end(self, outputs):
        self._log_dict(self._avg_dicts(outputs))

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.state = 'predict'
        outputs = self.forward(batch)
        return outputs

    def construct_lg(self, lg_prob, bsz, na_fea_type, max_lg_node):
        lg_atom_fea, lg_bond_adj, lg_dist_adj = torch.zeros(bsz, na_fea_type, max_lg_node), \
                                                torch.zeros(bsz, max_lg_node, max_lg_node), \
                                                torch.zeros(bsz, max_lg_node, max_lg_node)
        gate_token = torch.zeros(bsz, max_lg_node, device=device, dtype=torch.int)
        lg_id = self.greedy_search(lg_prob).detach().tolist()
        for idx in range(bsz):
            cur_lg = self.lg[lg_id[idx]]
            cur_na = int(cur_lg.na)
            lg_atom_fea[idx, :, :cur_na] = cur_lg.atom_fea
            lg_bond_adj[idx, :cur_na, :cur_na] = cur_lg.bond_adj
            lg_dist_adj[idx, :cur_na, :cur_na] = cur_lg.dist_adj
            for gi, gn in enumerate(cur_lg.gate_num):
                gate_token[idx, gi] = gn
        return lg_atom_fea, lg_bond_adj, lg_dist_adj

    def find_solutions(self, product, rc_adj_prob_normed, h_prob_normed, bond_type_adj, ct_probs, lg_prob_sorted,
                       lg_prob_indices, n_pro, device, ct_threshold=1e-2, rc_threshold=0.2, h_center_idx=3, start_k=0):
        solutions = []
        error_mol = []
        origin_numhs = []
        # rc_adj_prob_normed = rc_adj_prob[0, :n_pro, :n_pro]
        # h_prob_normed = h_prob.softmax(dim=-1)[0, :n_pro]
        # bond_type_adj = bond_fea2type(bond_adj)[0, :n_pro, :n_pro]
        # origin_numhs = dataset[pred_idx]['product']['atom_fea'][3,:n_pro] - 1

        rc_score_init = rc_adj_prob_normed
        rc_score_init = -(1 - rc_score_init).log().sum()
        h_prob_normed_loged = h_prob_normed[:n_pro].log()
        # product, reactant = get_reaction(pred_idx)
        # product = correct_charge(product)
        # print("rc_score_init:%.2f" % rc_score_init)
        for atom in product.GetAtoms():
            origin_numhs.append(atom.GetTotalNumHs())

        num_lg_consider = ct_probs.size(0)
        from tqdm import tqdm
        for kk in tqdm(range(start_k, start_k + num_lg_consider)):
            base_score = rc_score_init.clone()
            cur_lg = self.lg[lg_prob_indices[kk]]
            cur_pro = copy.deepcopy(product)
            cur_lg_adj = util.bond_fea2type(cur_lg.bond_adj)
            degree_change = [0 for _ in range(n_pro)]

            # build leaving_group
            for lg_atom_idx in range(cur_lg.na):
                # print(cur_lg.atom_fea[0, lg_atom_idx].item())
                cur_lg_atom = Chem.Atom(cur_lg.atom_fea[0, lg_atom_idx].int().item())
                cur_lg_atom.SetFormalCharge(cur_lg.atom_fea[6, lg_atom_idx].int().item() - 10)
                # cur_lg_atom.SetTotalNumHs(cur_lg.atom_fea[3, lg_atom_idx].item() - 1)
                cur_pro.AddAtom(cur_lg_atom)
                for lg_atom_idx_j in range(lg_atom_idx):
                    if cur_lg_adj[lg_atom_idx, lg_atom_idx_j] > 0:
                        cur_pro.AddBond(lg_atom_idx + n_pro, lg_atom_idx_j + n_pro,
                                        self.bond_decoder[float(cur_lg_adj[lg_atom_idx, lg_atom_idx_j])])

            # flag, atomid_valence = util.check_valency(cur_pro)
            # if not flag:
            #     assert len(atomid_valence) == 2
            #     idx = atomid_valence[0]
            #     v = atomid_valence[1]
            #     an = cur_pro.GetAtomWithIdx(idx).GetAtomicNum()
            #     if an in (7, 8, 16) and (v - util.ATOM_VALENCY[an]) == 1:
            #         cur_pro.GetAtomWithIdx(idx).SetFormalCharge(1)

            base_score += -lg_prob_sorted[kk].log()
            base_score += -(1 - ct_probs[kk - start_k] + 1e-30).log().sum()
            base_score += -(h_prob_normed_loged[:, h_center_idx].sum())
            actions = [f"LGM|Select Leaving Group with Index {lg_prob_indices[kk]} and Energy %.2f"
                       % -lg_prob_sorted[kk].log()]
            actions += [f"IT|Initial Energy:%.2f" % base_score]

            # media_mols = [copy.deepcopy(cur_pro)]

            #     cur_pro

            def dfs_for_ct(lg_atom_idx, pro_atom_idx, cur_gate_num, cur_score, depth=0):
                if lg_atom_idx >= len(cur_lg.gate_num):
                    dfs_for_rc(0, 0, cur_pro, cur_score)
                    return
                for atom_idx in range(pro_atom_idx, n_pro):
                    if ct_probs[kk - start_k, atom_idx, lg_atom_idx] > ct_threshold:
                        cur_ct_prob = ct_probs[kk - start_k, atom_idx, lg_atom_idx]
                        cur_gate_num += 1
                        if cur_gate_num <= cur_lg.gate_num[lg_atom_idx]:
                            dert_score = -(cur_ct_prob + 1e-30).log() + (1 - cur_ct_prob + 1e-30).log()
                            cur_bridge_idx = sum(cur_lg.gate_num[:lg_atom_idx]) + cur_gate_num - 1
                            try:
                                cur_pro.AddBond(atom_idx, lg_atom_idx + n_pro,
                                                self.bond_decoder[cur_lg.bridge[0][cur_bridge_idx][1]])
                                actions.append(
                                    f"RCP|Add Bonds: between {atom_idx} and {lg_atom_idx + n_pro} with Bond Type "
                                    f"{cur_lg.bridge[0][cur_bridge_idx][1]} and Cost %.2f" % dert_score)
                                # media_mols.append(copy.deepcopy(cur_pro))
                            except Exception as e:
                                # error_mol.append((copy.deepcopy(cur_pro), "ct_error", kk, e))
                                return
                            degree_change[atom_idx] += cur_lg.bridge[0][cur_bridge_idx][1]
                            # print("cur_score1: %.2f\t%.2f\tlg_atom_idx:%d\tpro_atom_idx:%d\tatom_idx:%d\tdepth:%d"
                            # %(cur_score, dert_score, lg_atom_idx, pro_atom_idx, atom_idx, depth))
                            if cur_gate_num == cur_lg.gate_num[lg_atom_idx]:
                                dfs_for_ct(lg_atom_idx + 1, 0, 0, cur_score + dert_score, depth + 1)
                            else:
                                dfs_for_ct(lg_atom_idx, pro_atom_idx + 1, cur_gate_num, cur_score + dert_score,
                                           depth + 1)
                            degree_change[atom_idx] -= cur_lg.bridge[0][cur_bridge_idx][1]
                            cur_pro.RemoveBond(atom_idx, lg_atom_idx + n_pro)
                            actions.pop()
                            # media_mols.pop()
                            cur_gate_num -= 1
                        else:
                            break

            def dfs_for_rc(start_row_idx, start_col_idx, mol, cur_score, depth=0):
                # print(lg_prob_indices[kk], depth, 'rc')
                # print("rc", depth, start_row_idx, start_col_idx)
                if depth > (rc_adj_prob_normed.sum() // 2):
                    return
                if start_col_idx >= n_pro:
                    start_col_idx = 0
                    start_row_idx += 1
                cur_mol = copy.deepcopy(mol)

                flag = False
                for atom_idx, atom in enumerate(cur_mol.GetAtoms()):
                    if atom_idx >= n_pro:
                        continue
                    if origin_numhs[atom_idx] < degree_change[atom_idx]:
                        formal_charge = int(degree_change[atom_idx] - origin_numhs[atom_idx])
                        if formal_charge < 2 and atom.GetAtomicNum() == 7:
                            atom.SetFormalCharge(formal_charge)
                        else:
                            break
                    if origin_numhs[atom_idx] < degree_change[atom_idx]:
                        break
                    else:
                        atom.SetNumExplicitHs(int(origin_numhs[atom_idx] - degree_change[atom_idx]))
                else:
                    try:
                        for atom in cur_mol.GetAtoms():
                            if atom.GetIsAromatic() and not atom.IsInRing():
                                atom.SetIsAromatic(False)
                        Chem.SanitizeMol(cur_mol)
                        flag = True
                    except Exception as e:
                        pass
                        # error_mol.append((copy.deepcopy(cur_mol), "rc_error", kk, e))

                if flag and len(actions) > 2:
                    dert_score = 0
                    cur_action = copy.deepcopy(actions)
                    for atom_idx, atom in enumerate(cur_mol.GetAtoms()):
                        if atom_idx < n_pro:
                            dert_h_num = atom.GetTotalNumHs() - origin_numhs[atom_idx]
                            if dert_h_num != 0:
                                if -3 <= dert_h_num <= 3:
                                    cur_dert_score = -h_prob_normed_loged[atom_idx, dert_h_num + h_center_idx] + (
                                    h_prob_normed_loged[atom_idx, h_center_idx])
                                else:
                                    cur_dert_score = 1000
                                dert_score += cur_dert_score
                                cur_action.append(
                                    f"H number change {dert_h_num} cost of atom {atom_idx + 1}: %.2f" % cur_dert_score)
                    # media_mols.append(cur_mol)
                    smiles = Chem.MolToSmiles(cur_mol)
                    if smiles is not None and Chem.MolFromSmiles(smiles) is not None:
                        solutions.append((smiles, float(cur_score + dert_score), cur_action, []))
                    print(solutions)

                for row_idx in range(start_row_idx, n_pro):
                    for col_idx in range(start_col_idx, n_pro):
                        if row_idx >= col_idx or rc_adj_prob_normed[row_idx, col_idx] < rc_threshold:
                            continue

                        for changed_bond in [0, 1, 2, 3]:
                            if changed_bond != bond_type_adj[row_idx, col_idx]:
                                dert_score = 0
                                degree_change[row_idx] += changed_bond - bond_type_adj[row_idx, col_idx]
                                degree_change[col_idx] += changed_bond - bond_type_adj[row_idx, col_idx]
                                cur_mol = copy.deepcopy(mol)
                                cur_mol.RemoveBond(row_idx, col_idx)
                                dert_score += -rc_adj_prob_normed[row_idx, col_idx].log() + (
                                        1 - rc_adj_prob_normed[row_idx, col_idx]).log()
                                if changed_bond != 0:
                                    cur_mol.AddBond(row_idx, col_idx, self.bond_decoder[changed_bond])
                                    actions.append(f"Replace Bonds: between {row_idx} and {col_idx}, " +
                                                   f"from Bond Type {int(bond_type_adj[row_idx, col_idx])} to Bond Type"
                                                   f"{changed_bond} with Cost %.2f" % dert_score)
                                else:
                                    actions.append(
                                        f"Remove Bonds: between {row_idx} and {col_idx}, with Bond Type "
                                        f"{bond_type_adj[row_idx, col_idx]} and Cost %.2f" % dert_score)

                                # media_mols.append(copy.deepcopy(cur_mol))
                                dfs_for_rc(row_idx, col_idx + 1, cur_mol, cur_score + dert_score, depth + 1)
                                actions.pop()
                                # media_mols.pop()
                                degree_change[row_idx] -= changed_bond - bond_type_adj[row_idx, col_idx]
                                degree_change[col_idx] -= changed_bond - bond_type_adj[row_idx, col_idx]

            dfs_for_ct(0, 0, 0, base_score)
        solutions.sort(key=lambda x: x[1])
        return solutions, []

    def run(self, smi, topk=10, max_num_atoms=150, max_num_lg_atoms=70, max_gate_num=10):

        # try:
        #     cur_smi = MultiStepDataset(smi, max_num_atoms=max_num_atoms, max_num_lg_atoms=max_num_lg_atoms)
        # except Exception as e:
        #     print(e)
        #     return {
        #     'reactants': [],
        #     'scores': [],
        #     'template': []
        # }
        try:
            cur_smi = MultiStepDataset(smi, max_num_atoms=max_num_atoms, max_num_lg_atoms=max_num_lg_atoms)
        except Exception as e:
            print(smi, e)
            return None
        cur_smi = DataLoader(cur_smi, batch_size=1)
        batch = next(iter(cur_smi))

        self.max_lg_consider = 1
        self.max_lg_node = max_num_lg_atoms
        self.max_gate_num = max_gate_num

        with torch.no_grad():
            pro_dic = batch['product']
            atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, = pro_dic['atom_fea'], pro_dic['bond_adj'], \
                                                                  pro_dic['dist_adj'], batch['center_cnt'], \
                                                                  batch['rxn_type']
            dist_adj_3d = pro_dic['dist_adj_3d'] if self.use_3d_info else None
            if not self.known_rxn_cnt:
                center_cnt = torch.zeros_like(center_cnt)

            device = batch['rxn_type'].device
            bsz, n_atom = 1, max_num_atoms
            na_fea_type = batch['product']['atom_fea'].size(1)

            if self.num_shared_layer > 0:
                shared_atom_fea, masked_adj = self.emb(atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist_adj_3d)
                shared_atom_fea = self.shared_encoder(shared_atom_fea, masked_adj)
                # rc_fea = self.rc_encoder(shared_atom_fea, masked_adj)
                rc_fea = shared_atom_fea
            else:
                shared_atom_fea, masked_adj = self.emb[0](atom_fea, bond_adj, dist_adj, center_cnt, rxn_type,
                                                          dist_adj_3d)
                rc_fea = self.rc_encoder(shared_atom_fea, masked_adj)

            rc_prob = self.rc_adj_fn(rc_fea[:, 1:], rc_fea[:, 1:], None)
            rc_adj_prob = self.rc_out_fn(rc_prob).squeeze() + torch.where(bond_adj > 1, 0., -torch.inf)
            h_fea = self.h_encoder(shared_atom_fea, masked_adj)
            h_prob = self.h_out_fn(h_fea[:, 1:])
            h_mask = torch.where(atom_fea[:, 0] > 0, 0., -1e3)
            h_prob[:, :, :3] += h_mask[:, :, None]
            h_prob[:, :, 4:] += h_mask[:, :, None]

            if self.num_shared_layer == 0:
                shared_atom_fea, masked_adj = self.emb[2](atom_fea, bond_adj, dist_adj, center_cnt, rxn_type,
                                                          dist_adj_3d)
            lg_fea = self.lg_encoder(shared_atom_fea, masked_adj)[:, 0]
            lg_prob = self.lg_out_fn(lg_fea)
            rc_adj_prob = rc_adj_prob.sigmoid()
            h_prob = h_prob.softmax(dim=-1)

            n_pro = batch['product']['n_atom'][0]
            # ct_probs = torch.zeros(topk, bsz, n_atom, max_gate_num, device=self.device)
            lg_prob_sorted, lg_prob_indices = lg_prob.softmax(dim=-1).sort(dim=-1, descending=True)

        # print(rc_adj_prob[:, :n_pro, :n_pro], h_prob[:, :n_pro, :n_pro], lg_prob_sorted[:, :10])
        bond_type_adj = util.bond_fea2type(bond_adj)
        solutions, sol_size = [], 0
        # from tqdm import tqdm
        for kk in range(topk * 10):
            # for kk in tqdm(range(topk * 10)):
            cur_lg_id = lg_prob_indices[:, kk]

            lg_atom_fea, lg_bond_adj, lg_dist_adj = torch.zeros(bsz, na_fea_type, self.max_lg_node, device=device), \
                                                    torch.zeros(bsz, self.max_lg_node, self.max_lg_node, device=device), \
                                                    torch.zeros(bsz, self.max_lg_node, self.max_lg_node, device=device)
            gate_token = torch.zeros(bsz, self.max_lg_node, dtype=torch.int, device=device)
            for idx in range(bsz):
                cur_lg = self.lg[cur_lg_id[idx]]
                cur_na = int(cur_lg.na)
                lg_atom_fea[idx, :, :cur_na] = cur_lg.atom_fea
                lg_bond_adj[idx, :cur_na, :cur_na] = cur_lg.bond_adj
                lg_dist_adj[idx, :cur_na, :cur_na] = cur_lg.dist_adj
                for gi, gn in enumerate(cur_lg.gate_num):
                    gate_token[idx, gi] = gn

            shared_atom_fea_lg2, masked_adj_lg2 = self.emb(lg_atom_fea, lg_bond_adj, lg_dist_adj,
                                                           center_cnt, rxn_type, dist3d_adj=None, contrast=True)
            # print(shared_atom_fea_lg2.device)
            shared_atom_fea_lg2[:, 1:] += self.gate_embedding(gate_token)
            ct_fea_lg2 = self.ct_encoder(shared_atom_fea_lg2, masked_adj_lg2)[:, 1:]
            # rc_fea = self.ct_encoder(shared_atom_fea, masked_adj)[:, 1:]
            ct_prob2 = self.ct_adj_fn(rc_fea[:, 1:], ct_fea_lg2)[:, :, :self.max_ct_atom]
            ct_prob2 = self.ct_out_fn(ct_prob2).squeeze() + torch.where(atom_fea[:, 0] > 0, 0., -1e3)[:, :, None]
            ct_prob2 = ct_prob2.sigmoid()

            i = 0
            cur_sols, error_mol = self.find_solutions(product=Chem.RWMol(Chem.MolFromSmiles(smi)),
                                                      rc_adj_prob_normed=rc_adj_prob[i, :n_pro, :n_pro],
                                                      h_prob_normed=h_prob[i, :n_pro],
                                                      bond_type_adj=bond_type_adj[i, :n_pro, :n_pro],
                                                      ct_probs=ct_prob2[:, :n_pro, :self.max_gate_num],
                                                      lg_prob_sorted=lg_prob_sorted[i],
                                                      lg_prob_indices=lg_prob_indices[i],
                                                      n_pro=int(n_pro),
                                                      device=device,
                                                      rc_threshold=0.5,
                                                      ct_threshold=0.5,
                                                      start_k=kk)
            sol_size += len(cur_sols)
            if cur_sols:
                solutions.extend(copy.deepcopy(cur_sols))
            if sol_size > topk:
                break
        solutions.sort(key=lambda x: x[1])
        return {
            'reactants': [solutions[i][0] for i in range(sol_size)],
            'scores': [float(solutions[i][1]) for i in range(sol_size)],
            'template': [solutions[i][2] for i in range(sol_size)]
        }

    @staticmethod
    def greedy_search(prob):
        """
        :param prob: [bsz, n_atom, n_atom, n_bond] if search_lg else [bsz, len(leaving_group)]
        :return:[bsz, n_atom, n_atom] if search_lg else [bsz]
        """
        return prob.argmax(dim=-1)

    @staticmethod
    def get_rxn_center(adj1: Tensor, adj2: Tensor):
        """
        :param adj1: [bsz, n, n]
        :param adj2: [bsz, n, n]
        :return: rxn_center idx [ins_idx, center_idx]
        """
        return (adj1 - adj2).nonzero()[:, :2]

    def preprocess_for_acc(self, batch, rc_prob, lg_prob, ct_prob, h_prob, greedy=True):
        rxn_cnt, rxn_type = batch['center_cnt'], batch['rxn_type']
        rc_prob = torch.sigmoid(rc_prob)
        # rc_h_prob = self.greedy_search(rc_h_prob)
        h_pred = self.greedy_search(h_prob)
        # h_pred = h_prob.round()
        bsz = rxn_cnt.size(0)
        if self.known_rxn_type:
            for idx in range(bsz):
                excluded_idx = self.lg.get_excluded_idxs(rxn_cnt[idx], rxn_type[idx])
                lg_prob[idx, excluded_idx] -= torch.inf
        elif self.known_rxn_cnt:
            for idx in range(bsz):
                excluded_idx = self.lg.get_excluded_idxs(rxn_cnt[idx], None)
                lg_prob[idx, excluded_idx] -= torch.inf
        if greedy:
            lg_pred = self.greedy_search(lg_prob)
            return rc_prob, lg_pred, ct_prob, h_pred
        else:
            n_pros, rc_target, lg_truth = batch['product']['n_atom'], batch['rc_target'], batch['lg_id']
            lg_pred = [self.find_lg_topk(lg_prob[idx], k=10, lg_truth=lg_truth[idx]) for idx in range(bsz)]
            rc_pred = [self.find_rc_topk(torch.triu(rc_prob[idx, :n_pros[idx], :n_pros[idx]], diagonal=1),
                                         k=10, rxn_cnt=rxn_cnt[idx], n_pro=n_pros[idx], rc=rc_target[idx]) for idx in
                       range(bsz)]
            mix_pred = []
            for idx in range(bsz):
                cur_mix_pred = []
                for i in range(10):
                    for j in range(10):
                        cur_mix_pred.append([lg_pred[idx][i][0] + rc_pred[idx][j][0],
                                             lg_pred[idx][i][1] and rc_pred[idx][j][1]])
                        if lg_pred[idx][i][1] and rc_pred[idx][j][1]:
                            break
                cur_mix_pred = sorted(cur_mix_pred, key=lambda x: x[0], reverse=True)
                ks = [11, 11, 101]
                k_list = [10, 10, 100]
                for i, candidate in enumerate([lg_pred[idx], rc_pred[idx], cur_mix_pred]):
                    for k in range(k_list[i]):
                        if candidate[k][1]:
                            ks[i] = k + 1
                            break
                lg_pred[idx], rc_pred[idx] = ks[0], ks[1]
                mix_pred.append(ks[0] * ks[1])
            return rc_pred, lg_pred, mix_pred, rc_prob, h_pred

    def find_lg_topk(self, cur_prob, k=10, lg_truth=None):
        values, indexes = torch.topk(cur_prob, k=k)
        values = - torch.arange(0, k, dtype=torch.float32, device=cur_prob.device)
        values = values.tolist()
        is_lg = torch.eq(indexes, lg_truth)
        return [(values[idx], bool(is_lg[idx])) for idx in range(k)]

    def find_rc_topk(self, cur_prob, k=10, rxn_cnt=None, n_pro=None, rc=None):
        if rxn_cnt == 0:
            return [(1., True)] + [(-1000., False) for _ in range(k - 1)]
        rc_truth = rc.nonzero()
        rc_truth = set([int(r[0] * n_pro + r[1]) for r in rc_truth if r[0] < r[1]])
        topk_size = max(k, int(rxn_cnt))
        values, indexes = torch.topk(cur_prob.view(n_pro * n_pro), k=topk_size)
        values, indexes = values.tolist(), indexes.tolist()
        # self.print(len(values), rxn_cnt)
        value = sum(values[:int(rxn_cnt)])
        comb = [i for i in range(rxn_cnt)]
        stack = []
        heapq.heappush(stack, (-value, comb))
        res, kk = [], 0

        while stack:
            value, comb = heapq.heappop(stack)
            value = -value
            # print([indexes[com] for com in comb], rc_truth, value)
            cur_pred = set([indexes[com] for com in comb])
            is_rc = bool(cur_pred == rc_truth)
            res.append((value, is_rc, cur_pred))
            kk += 1
            if kk == k:
                break
            for center_idx in range(rxn_cnt - 1, -1, -1):
                for i in range(comb[center_idx] + 1, topk_size):
                    cur_value = value + values[i] - values[comb[center_idx]]
                    cur_comb = comb.copy()
                    cur_comb[center_idx] = i
                    heapq.heappush(stack, (-cur_value, cur_comb))
        values = torch.tensor([-idx for idx, r in enumerate(res)])
        res = [(values[i], res[i][1], res[i][2]) for i in range(kk)]
        for i in range(kk, k + 1):
            res.append((-10000, False, []))
        return res

    def calc_batch_accuracy(self, batch, rc_prob, lg_prob, ct_prob, h_prob):
        rc_prob, lg_pred, ct_prob, h_pred = \
            self.preprocess_for_acc(batch, rc_prob, lg_prob.clone(), ct_prob, h_prob, greedy=True)
        rc_truth, lg_truth, rxn_cnt = batch['rc_target'], batch['lg_id'], batch['center_cnt']
        rc_h_truth = batch['rc_h']
        ct_truth = batch['ct_target']
        n_pros = batch['product']['n_atom']

        all_cnt, rc_cnt, lg_cnt, rc_atom_cnt, ct_cnt, h_cnt = 0, 0, 0, 0, 0, 0
        bsz_size = rc_prob.size(0)
        for idx, n_pro in enumerate(n_pros):
            k = rxn_cnt[idx]
            cur_rc_prob = torch.triu(rc_prob[idx, :n_pro, :n_pro], diagonal=1)
            topk_value = torch.topk(cur_rc_prob.view(n_pro * n_pro), k=k).values[-1] if k > 0 else 1
            cur_rc_pred = torch.where(cur_rc_prob >= topk_value, 1., 0.)
            is_rc_eq = int(
                torch.equal(cur_rc_pred, torch.triu(rc_truth[idx, :n_pro, :n_pro], diagonal=1)))

            cur_h_pred = h_pred[idx, :n_pro]
            # if k > 0:
            #     h_pred -= 1
            #     h_pred *= torch.where(cur_rc_pred.sum(dim=0) > 0, 1, 0)
            #     h_pred += 1
            #
            # rc_h_atom_cnt += torch.eq(h_pred, rc_h_truth[idx, :n_pro]).float().mean()
            is_h_eq = int(torch.equal(cur_h_pred, rc_h_truth[idx, :n_pro]))
            h_cnt += is_h_eq

            is_ct_eq = 1
            cur_ct_preds = ct_prob[idx, :n_pro]
            gate_num = self.lg[lg_truth[idx]].gate_num
            for g_idx in range(len(gate_num) - 1):
                k = gate_num[g_idx]
                topk_value = torch.topk(cur_ct_preds[:, g_idx], k=k).values[-1] if k > 0 else 1
                cur_ct_pred = torch.where(cur_ct_preds[:, g_idx] >= topk_value, 1., 0.)
                is_ct_eq *= int(torch.equal(cur_ct_pred, ct_truth[idx, :n_pro, g_idx]))
                if is_ct_eq == 0:
                    break

            is_lg_eq = int(lg_pred[idx] == lg_truth[idx])

            rc_cnt += is_rc_eq
            lg_cnt += is_lg_eq
            ct_cnt += is_ct_eq
            all_cnt += is_rc_eq * is_lg_eq * is_ct_eq * is_h_eq
        return all_cnt / bsz_size, rc_cnt / bsz_size, lg_cnt / bsz_size, ct_cnt / bsz_size, h_cnt / bsz_size

    def calc_beam_search_accuracy(self, batch, rc_prob, lg_prob, ct_prob, h_prob):
        rc_pred, lg_pred, mix_pred, _, _ = \
            self.preprocess_for_acc(batch, rc_prob, lg_prob.clone(), ct_prob, h_prob, greedy=False)

        all_acc = {f"all_top{k}_acc": 0 for k in [1, 3, 5, 10]}
        rc_acc = {f"rc_top{k}_acc": 0 for k in [1, 3, 5, 10]}
        lg_acc = {f"lg_top{k}_acc": 0 for k in [1, 3, 5, 10]}
        bsz = rc_prob.size(0)

        all_acc['all_top1_acc'], rc_acc['rc_top1_acc'], lg_acc['lg_top1_acc'], ct_acc, h_acc = \
            self.calc_batch_accuracy(batch, rc_prob, lg_prob, ct_prob, h_prob)

        if '50k' in self.dataset_path and 'tsplit' not in self.dataset_path and False: 
            rxn_type = batch['rxn_type']
            rxn_type_acc = {f"rxn_type{i}_top{k}_acc": 0 for i in range(10) for k in [0, 1, 3, 5, 10]}

            for idx in range(bsz):
                cur_rxn_type = rxn_type[idx].int().item()
                rxn_type_acc[f"rxn_type{cur_rxn_type}_top{0}_acc"] += 1
                if mix_pred[idx] <= 1:
                    rxn_type_acc[f"rxn_type{cur_rxn_type}_top{1}_acc"] += 1
                for k in [3, 5, 10]:
                    if rc_pred[idx] <= k:
                        rc_acc[f'rc_top{k}_acc'] += 1
                    if lg_pred[idx] <= k:
                        lg_acc[f'lg_top{k}_acc'] += 1
                    if mix_pred[idx] <= k:
                        all_acc[f"all_top{k}_acc"] += 1
                        rxn_type_acc[f"rxn_type{cur_rxn_type}_top{k}_acc"] += 1

            for k in [3, 5, 10]:
                rc_acc[f'rc_top{k}_acc'] /= bsz
                lg_acc[f'lg_top{k}_acc'] /= bsz
                all_acc[f'all_top{k}_acc'] /= bsz
            for k in [1, 3, 5, 10]:
                for i in range(10):
                    if rxn_type_acc[f"rxn_type{i}_top{0}_acc"] > 0:
                        rxn_type_acc[f"rxn_type{i}_top{k}_acc"] /= rxn_type_acc[f"rxn_type{i}_top{0}_acc"]
                    else:
                        rxn_type_acc[f"rxn_type{i}_top{k}_acc"] = None

        else:
            rxn_type_acc = None
            for k in [3, 5, 10]:
                for idx in range(bsz):
                    if rc_pred[idx] <= k:
                        rc_acc[f'rc_top{k}_acc'] += 1
                    if lg_pred[idx] <= k:
                        lg_acc[f'lg_top{k}_acc'] += 1
                    if mix_pred[idx] <= k:
                        all_acc[f"all_top{k}_acc"] += 1
            for k in [3, 5, 10]:
                rc_acc[f'rc_top{k}_acc'] /= bsz
                lg_acc[f'lg_top{k}_acc'] /= bsz
                all_acc[f'all_top{k}_acc'] /= bsz

        return all_acc, rc_acc, lg_acc, ct_acc, h_acc, rxn_type_acc

    @staticmethod
    def _avg_dicts(colls):
        complete_dict = {}
        for coll in colls:
            for key, val in coll.items():
                if key not in complete_dict:
                    complete_dict[key] = []
                complete_dict[key].append(val)
        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--nhead', type=int, default=16)
        parser.add_argument('--num_shared_layer', type=int, default=6)
        parser.add_argument('--num_rc_layer', type=int, default=1)
        parser.add_argument('--num_lg_layer', type=int, default=1)
        parser.add_argument('--num_h_layer', type=int, default=1)
        parser.add_argument('--num_ct_layer', type=int, default=1)
        parser.add_argument('--n_rxn_type', type=int, default=10)
        parser.add_argument('--n_rxn_cnt', type=int, default=100)
        parser.add_argument('--dim_feedforward', type=int, default=2048)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--max_ct_atom', type=int, default=4)
        parser.add_argument('--batch_second', default=False, action='store_true')
        parser.add_argument('--known_rxn_type', default=False, action='store_true')
        # parser.add_argument('--use_center', default=False, action='store_true')
        parser.add_argument('--not_known_rxn_cnt', default=False, action='store_true')
        parser.add_argument('--norm_first', default=False, action='store_true')
        parser.add_argument('--activation', type=str, default='gelu')
        parser.add_argument("--warmup_updates", type=int, default=6000)
        parser.add_argument("--tot_updates", type=int, default=200000)
        parser.add_argument("--peak_lr", type=float, default=1e-4)
        parser.add_argument("--end_lr", type=float, default=1e-7)
        parser.add_argument('--weight_decay', type=float, default=1e-2)
        parser.add_argument('--max_single_hop', type=int, default=4)
        parser.add_argument('--not_use_dist_adj', default=False, action='store_true')
        parser.add_argument('--not_use_contrastive', default=False, action='store_true')
        parser.add_argument('--not_use_adaptive_multi_task', default=False, action='store_true')
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay
        )
        lr_scheduler = {
            "scheduler": PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
