# !/usr/bin/python3
# @File: LeavingGroup.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.24.18
import torch


class LeavingGroup:
    def __init__(self, na, atom_fea, bond_adj, gate_num, center_cnt, rxn_type, bridge, dist_adj=None):
        self.na = na
        self.atom_fea = atom_fea
        self.bond_adj = bond_adj
        self.rxn_type = rxn_type
        self.gate_num = gate_num
        self.center_cnt = center_cnt
        self.bridge = bridge
        self.dist_adj = dist_adj
        self.n = 1

    def __eq__(self, other):
        return isinstance(other, type(self)) \
               and self.na == other.na \
               and self.gate_num == other.gate_num \
               and torch.equal(self.atom_fea[0, :], other.atom_fea[0, :]) \
               and torch.equal(self.bond_adj, other.bond_adj) \
               and all([self.bridge[0][i][1] == other.bridge[0][i][1] for i in range(sum(self.gate_num))])
