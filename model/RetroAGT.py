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
                 atom_dim=90, total_degree=10, formal_charge=8, hybrid=7, exp_valance=8, hydrogen=10, aromatic=2,
                 ring=10, n_layers=1, batch_first=True, known_rxn_type=True, known_rxn_cnt=True, norm_first=False,
                 activation='gelu', warmup_updates=6e4, tot_updates=1e6, peak_lr=2e-4, end_lr=1e-9, weight_decay=0.99,
                 leaving_group_path=None, use_3d_info=False, use_dist_adj=True, dataset_path=None):
        super().__init__()
        assert os.path.isfile(leaving_group_path)
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
        self.lg = LeavingGroupList(torch.load(leaving_group_path), n_rxn_type, n_rxn_cnt)
        self.lg_size = len(self.lg)
        self.batch_size = None

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
            nn.Linear(d_model, self.lg_size),
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

        self.loss_init = None
        self.loss_last = None
        self.loss_last2 = None
        self.max_lg_consider = 10
        self.max_lg_node = 30
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

            shared_atom_fea_lg2[:, 1:] += self.gate_embedding(gate_token)
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

        ct_fea_lg = self.ct_encoder(shared_atom_fea_lg, masked_adj_lg)[:, 1:]
        # rc_fea = self.ct_encoder(shared_atom_fea, masked_adj)[:, 1:]
        ct_prob = self.ct_adj_fn(rc_fea[:, 1:], ct_fea_lg, None)[:, :, :self.max_ct_atom]
        ct_prob = self.ct_out_fn(ct_prob).squeeze() + torch.where(atom_fea[:, 0] > 0, 0., -1e3)[:, :, None]
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
                "rxn_type": batch['rxn_type'],
                "lg_id": batch['lg_id'],
                # "lg_id_pred": self.greedy_search(lg_prob),
                "rc_adj_prob": rc_adj_prob,
                "lg_fea": lg_fea,
                "lg_fea2": lg_fea2,
                "lg_prob": lg_prob,
                # "lg_prob2": lg_prob2,
                "ct_prob_real": ct_prob,
                "ct_prob_pred": ct_prob2,
            }

        return rc_adj_prob, lg_prob, lg_prob2, ct_prob, h_prob

    def calc_loss(self, batch, rc_adj_prob, lg_prob, lg_prob2, ct_prob, h_prob):
        bsz, n_atom, _ = rc_adj_prob.size()
        loss_rc = self.criterion_rc(rc_adj_prob.reshape(bsz * n_atom * n_atom),
                                    batch["rc_target"].reshape(bsz * n_atom * n_atom))
        loss_voc = self.criterion_lg(lg_prob, batch['lg_id'])

        if self.use_contrastive:
            loss_voc += self.criterion_lg(lg_prob2, batch['lg_id'])
            loss_voc /= 2
        loss_ct = self.criterion_ct(ct_prob, batch['ct_target'])
        loss_h = self.criterion_h(h_prob.reshape(bsz * n_atom, -1), batch['rc_h'].reshape(bsz * n_atom).long())

        return loss_rc, loss_voc, loss_ct, loss_h

    def calc_mt_loss(self, loss_list):
        loss_list = torch.stack(loss_list)
        if not self.use_adaptive_multi_task or self.num_shared_layer == 0:
            return loss_list.sum()

        if self.loss_init is None:
            if self.training:
                self.loss_init = loss_list.detach()
                self.loss_last2 = loss_list.detach()
                loss_t = (loss_list / self.loss_init).mean()
            else:
                loss_t = (loss_list / loss_list.detach()).mean()

        elif self.loss_last is None:
            if self.training:
                self.loss_last = loss_list.detach()
                loss_t = (loss_list / self.loss_init).mean()
            else:
                loss_t = (loss_list / loss_list.detach()).mean()
        else:
            w = F.softmax(self.loss_last / self.loss_last2, dim=-1).detach()
            loss_t = (loss_list / self.loss_init * w).sum()

            if self.training:
                self.loss_last2 = self.loss_last
                self.loss_last = loss_list.detach()
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
        all_acc, rc_acc, lg_acc, ct_acc, h_acc = \
            self.calc_beam_search_accuracy(batch, rc_adj_prob, lg_prob, ct_prob, h_prob)
        test_output = {}

        for k in [1, 3, 5, 10]:
            test_output[f"all_top{k}_beam_acc"] = all_acc[f"all_top{k}_acc"]
            test_output[f"lg_top{k}_beam_acc"] = lg_acc[f"lg_top{k}_acc"]
            test_output[f"rc_top{k}_beam_acc"] = rc_acc[f"rc_top{k}_acc"]
            test_output[f"all_top{k}_acc"] = 0

        test_output['ct_acc'] = ct_acc
        test_output['h_acc'] = h_acc

        if self.dataset is None:
            self.dataset = pd.read_csv(os.path.join(self.dataset_path, "processed/test/smiles_lists.csv"))

        pro_dic = batch['product']
        atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, = pro_dic['atom_fea'], pro_dic['bond_adj'], \
                                                              pro_dic['dist_adj'], batch['center_cnt'], \
                                                              batch['rxn_type']
        dist_adj_3d = pro_dic['dist_adj_3d'] if self.use_3d_info else None
        if not self.known_rxn_cnt:
            center_cnt = torch.zeros_like(center_cnt)
        bsz, n_atom, _ = bond_adj.size()
        bsz, na_fea_type, _ = pro_dic['atom_fea'].size()
        if self.batch_size is None:
            self.batch_size = bsz

        if self.num_shared_layer > 0:
            shared_atom_fea, masked_adj = self.emb(atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist_adj_3d)
            shared_atom_fea = self.shared_encoder(shared_atom_fea, masked_adj)
            # rc_fea = self.rc_encoder(shared_atom_fea, masked_adj)
            rc_fea = shared_atom_fea
        else:
            shared_atom_fea, masked_adj = self.emb[0](atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist_adj_3d)
            rc_fea = self.rc_encoder(shared_atom_fea, masked_adj)

        device = batch['rxn_type'].device

        bsz, n_atom, _ = batch['product']['bond_adj'].size()
        bsz, na_fea_type, _ = batch['product']['atom_fea'].size()

        ct_probs = torch.zeros(self.max_lg_consider, bsz, n_atom, self.max_gate_num, device=device)
        lg_prob_sorted, lg_prob_indices = lg_prob.softmax(dim=-1).sort(dim=-1, descending=True)

        bond_type_adj = util.bond_fea2type(bond_adj)
        for kk in range(self.max_lg_consider):
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
            ct_probs[kk] = ct_prob2.sigmoid()

        n_pros = batch['product']['n_atom']
        rc_adj_prob = rc_adj_prob.sigmoid()
        h_prob = h_prob.softmax(dim=-1)

        cur_bsz = bsz
        for i in range(bsz):
            n_pro = n_pros[i]
            if center_cnt[i] > 10:
                cur_bsz -= 1
                continue
            global_idx = self.batch_size * batch_idx + i
            # print(global_idx)
            reactant_smi, product_smi = self.dataset.iloc[global_idx, 0], self.dataset.iloc[global_idx, 1]
            product = Chem.MolFromSmiles(product_smi)
            product_smi = Chem.MolToSmiles(product)
            reactant_smi = Chem.MolToSmiles(Chem.MolFromSmiles(reactant_smi))
            assert product.GetNumAtoms() == n_pro
            solutions, error_mol = self.find_solutions(product=Chem.RWMol(product),
                                                       rc_adj_prob_normed=rc_adj_prob[i, :n_pro, :n_pro],
                                                       h_prob_normed=h_prob[i, :n_pro],
                                                       bond_type_adj=bond_type_adj[i, :n_pro, :n_pro],
                                                       ct_probs=ct_probs[:, i, :n_pro, :self.max_gate_num],
                                                       lg_prob_sorted=lg_prob_sorted[i],
                                                       lg_prob_indices=lg_prob_indices[i],
                                                       n_pro=int(n_pro),
                                                       device=device
                                                       )
            # print(global_idx, len(solutions), len(error_mol))
            if len(solutions) < 10:
                print(global_idx)
                cur_bsz -= 1
                continue
            for k in range(min(len(solutions), 10)):
                # cur_smiles = Chem.MolToSmiles(solutions[k][0])
                cur_smiles = solutions[k][0]
                # print(solutions[k][1], solutions[k][2])
                if cur_smiles == product_smi:
                    continue
                if cur_smiles == reactant_smi:
                    for kk in [1, 3, 5, 10]:
                        if k <= kk:
                            test_output[f"all_top{kk}_acc"] += 1
                    break
            for kk in [1, 3, 5, 10]:
                test_output[f"all_top{kk}_acc"] /= cur_bsz
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
        # product, reactant = get_reaction(pred_idx)
        # product = correct_charge(product)
        # print("rc_score_init:%.2f" % rc_score_init)
        for atom in product.GetAtoms():
            origin_numhs.append(atom.GetTotalNumHs())

        num_lg_consider = ct_probs.size(0)
        for kk in range(start_k, start_k+num_lg_consider):
            base_score = rc_score_init.clone()
            cur_lg = self.lg[lg_prob_indices[kk]]
            cur_pro = copy.deepcopy(product)
            cur_lg_adj = util.bond_fea2type(cur_lg.bond_adj)
            degree_change = [0 for _ in range(n_pro)]

            # build leaving_group
            for lg_atom_idx in range(cur_lg.na):
                cur_lg_atom = Chem.Atom(int(cur_lg.atom_fea[0, lg_atom_idx]))
                if int(cur_lg.atom_fea[0, lg_atom_idx]) == 7 and cur_lg.atom_fea[1, lg_atom_idx] == 4 and \
                        cur_lg.atom_fea[2, lg_atom_idx] == 4:
                    cur_lg_atom.SetFormalCharge(1)
                cur_pro.AddAtom(cur_lg_atom)
                for lg_atom_idx_j in range(lg_atom_idx):
                    if cur_lg_adj[lg_atom_idx, lg_atom_idx_j] > 0:
                        cur_pro.AddBond(lg_atom_idx + n_pro, lg_atom_idx_j + n_pro,
                                        self.bond_decoder[float(cur_lg_adj[lg_atom_idx, lg_atom_idx_j])])

            flag, atomid_valence = util.check_valency(cur_pro)
            if not flag:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = cur_pro.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - util.ATOM_VALENCY[an]) == 1:
                    cur_pro.GetAtomWithIdx(idx).SetFormalCharge(1)

            base_score += -lg_prob_sorted[kk].log()
            base_score += -(1 - ct_probs[kk-start_k]).log().sum()
            actions = [f"Select Leaving Group with Index {lg_prob_indices[kk]} and Cost %.2f"
                       % -lg_prob_sorted[kk].log()]
            actions += [f"Initial Cost:%.2f" % base_score]

            # media_mols = [copy.deepcopy(cur_pro)]

            #     cur_pro
            def dfs_for_ct(lg_atom_idx, pro_atom_idx, cur_gate_num, cur_score, depth=0):
                if lg_atom_idx >= len(cur_lg.gate_num):
                    dfs_for_rc(0, 0, cur_pro, cur_score)
                    return
                for atom_idx in range(pro_atom_idx, n_pro):
                    if ct_probs[kk-start_k, atom_idx, lg_atom_idx] > ct_threshold:
                        cur_ct_prob = ct_probs[kk-start_k, atom_idx, lg_atom_idx]
                        cur_gate_num += 1
                        if cur_gate_num <= cur_lg.gate_num[lg_atom_idx]:
                            dert_score = -cur_ct_prob.log() + (1 - cur_ct_prob).log()
                            cur_bridge_idx = sum(cur_lg.gate_num[:lg_atom_idx]) + cur_gate_num - 1
                            try:
                                cur_pro.AddBond(atom_idx, lg_atom_idx + n_pro,
                                                self.bond_decoder[cur_lg.bridge[0][cur_bridge_idx][1]])
                                actions.append(
                                    f"Add Bonds: between {atom_idx} and {lg_atom_idx + n_pro} with Bond Type "
                                    f"{cur_lg.bridge[0][cur_bridge_idx][1]} and Cost %.2f" % dert_score)
                                # media_mols.append(copy.deepcopy(cur_pro))
                            except Exception as e:
                                error_mol.append((copy.deepcopy(cur_pro), "ct_error", kk, e))
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
                if depth > 3:
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
                        break
                    else:
                        atom.SetNumExplicitHs(int(origin_numhs[atom_idx] - degree_change[atom_idx]))
                else:
                    try:
                        for atom in cur_mol.GetAtoms():
                            if atom.GetIsAromatic() and not atom.IsInRing():
                                atom.SetAromatic(False)
                        Chem.SanitizeMol(cur_mol)
                        flag = True
                    except Exception as e:
                        error_mol.append((copy.deepcopy(cur_mol), "rc_error", kk, e))

                if flag and len(actions) > 2:
                    dert_score = 0
                    cur_action = copy.deepcopy(actions)
                    for atom_idx, atom in enumerate(cur_mol.GetAtoms()):
                        if atom_idx >= n_pro:
                            continue
                        dert_h_num = atom.GetTotalNumHs() - origin_numhs[atom_idx]
                        if dert_h_num != 0 and -h_center_idx <= dert_h_num <= h_center_idx:
                            cur_dert_score = -h_prob_normed[atom_idx, dert_h_num + h_center_idx].log()
                            dert_score += cur_dert_score
                            cur_action.append(
                                f"H number change {dert_h_num} cost of atom {atom_idx + 1}: %.2f" % cur_dert_score)
                    # media_mols.append(cur_mol)
                    smiles = Chem.MolToSmiles(cur_mol)
                    solutions.append((smiles, float(cur_score + dert_score), cur_action, []))

                for row_idx in range(start_row_idx, n_pro):
                    for col_idx in range(start_col_idx, n_pro):
                        if row_idx >= col_idx or rc_adj_prob_normed[row_idx, col_idx] < rc_threshold:
                            continue

                        for changed_bond in [0, 1, 1.5, 2, 3]:
                            degree_change[row_idx] += changed_bond - bond_type_adj[row_idx, col_idx]
                            degree_change[col_idx] += changed_bond - bond_type_adj[row_idx, col_idx]

                            dert_score = 0
                            if changed_bond != bond_type_adj[row_idx, col_idx]:
                                cur_mol = copy.deepcopy(mol)
                                cur_mol.RemoveBond(row_idx, col_idx)
                                dert_score += -rc_adj_prob_normed[row_idx, col_idx].log() + (
                                        1 - rc_adj_prob_normed[row_idx, col_idx]).log()
                                if changed_bond != 0:
                                    cur_mol.AddBond(row_idx, col_idx, self.bond_decoder[changed_bond])
                                    actions.append(f"Replace Bonds: between {row_idx} and {col_idx}, " +
                                                   f"from Bond Type {bond_type_adj[row_idx, col_idx]} to Bond Type"
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
        return solutions, error_mol

    def run(self, smi, topk=10, max_num_atoms=150, max_num_lg_atoms=70, max_gate_num=10):

        # print("Query smiles:", smi)
        cur_smi = MultiStepDataset(smi, max_num_atoms=max_num_atoms, max_num_lg_atoms=max_num_lg_atoms)
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
                shared_atom_fea, masked_adj = self.emb[0](atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist_adj_3d)
                rc_fea = self.rc_encoder(shared_atom_fea, masked_adj)

            rc_prob = self.rc_adj_fn(rc_fea[:, 1:], rc_fea[:, 1:], None)
            rc_adj_prob = self.rc_out_fn(rc_prob).squeeze() + torch.where(bond_adj > 1, 0., -torch.inf)
            h_fea = self.h_encoder(shared_atom_fea, masked_adj)
            h_prob = self.h_out_fn(h_fea[:, 1:])
            h_mask = torch.where(atom_fea[:, 0] > 0, 0., -1e3)
            h_prob[:, :, :3] += h_mask[:, :, None]
            h_prob[:, :, 4:] += h_mask[:, :, None]

            if self.num_shared_layer == 0:
                shared_atom_fea, masked_adj = self.emb[2](atom_fea, bond_adj, dist_adj, center_cnt, rxn_type, dist_adj_3d)
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


            i=0
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
                        cur_mix_pred.append((lg_pred[idx][i][0] + rc_pred[idx][j][0] * 10,
                                             lg_pred[idx][i][1] and rc_pred[idx][j][1]))
                        if lg_pred[idx][i][1] and rc_pred[idx][j][1]:
                            break
                cur_mix_pred = sorted(cur_mix_pred, key=lambda x: x[0], reverse=True)
                ks = [101, 101, 101]
                k_list = [10, 10, 100]
                for i, candidate in enumerate([lg_pred[idx], rc_pred[idx], cur_mix_pred]):
                    for k in range(k_list[i]):
                        if candidate[k][1]:
                            ks[i] = k + 1
                            break
                lg_pred[idx], rc_pred[idx] = ks[0], ks[1]
                mix_pred.append(ks[2])
                # print(ks[0], ks[1], ks[2])
            return rc_pred, lg_pred, mix_pred, rc_prob, h_pred

    def find_lg_topk(self, cur_prob, k=10, lg_truth=None):
        values, indexes = torch.topk(cur_prob, k=k)
        values = F.softmax(values, dim=-1)
        values = values.tolist()
        is_lg = torch.eq(indexes, lg_truth)
        return [(values[idx], bool(is_lg[idx])) for idx in range(k)]

    def find_rc_topk(self, cur_prob, k=10, rxn_cnt=None, n_pro=None, rc=None):
        if rxn_cnt == 0:
            return [(1., True)] + [(0., False) for _ in range(k - 1)]
        rc_truth = rc.nonzero()
        rc_truth = [int(r[0] * n_pro + r[1]) for r in rc_truth if r[0] < r[1]]
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
            is_rc = bool([indexes[com] for com in comb] == rc_truth)
            res.append((value, is_rc))
            kk += 1
            if kk == k:
                break
            for center_idx in range(rxn_cnt - 1, -1, -1):
                for i in range(comb[center_idx] + 1, topk_size):
                    cur_value = value + values[i] - values[comb[center_idx]]
                    cur_comb = comb.copy()
                    cur_comb[center_idx] = i
                    heapq.heappush(stack, (-cur_value, cur_comb))
        values = torch.tensor([r[0] for r in res]).softmax(dim=-1)
        res = [(values[i], res[i][1]) for i in range(kk)]
        for i in range(kk, k + 1):
            res.append((0., False))
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

        return all_acc, rc_acc, lg_acc, ct_acc, h_acc

    @staticmethod
    def _avg_dicts(colls):
        complete_dict = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]
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
        parser.add_argument('--not_known_rxn_cnt', default=True, action='store_false')
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
