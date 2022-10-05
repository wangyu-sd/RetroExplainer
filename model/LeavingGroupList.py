# !/usr/bin/python
# @File: LeavingGroupList.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022/4/21 21:03
# @Software: PyCharm
from data.LeavingGroup import LeavingGroup


class LeavingGroupList:
    def __init__(self, leaving_group_list, n_rxn_type=10, n_rxn_cnt=7):
        self.lg = leaving_group_list
        self.n_rxn_type = n_rxn_type
        self.n_rxn_cnt = n_rxn_cnt

        self.excluded_rc_cnt = [[] for _ in range(self.n_rxn_cnt)]
        self.excluded_rxn_type = [[] for _ in range(self.n_rxn_type)]
        self.excluded_idx = [[[] for _ in range(self.n_rxn_type)] for _ in range(self.n_rxn_cnt)]
        self.lg_cnt_sum = 0

        for idx, l in enumerate(self.lg):

            self.lg_cnt_sum += l.n

            for cnt in range(self.n_rxn_cnt):
                if cnt not in l.center_cnt:
                    self.excluded_rc_cnt[cnt].append(idx)
                    for r_t in range(self.n_rxn_type):
                        self.excluded_idx[cnt][r_t].append(idx)
                else:
                    for r_t in range(self.n_rxn_type):
                        if r_t not in l.rxn_type:
                            self.excluded_idx[cnt][r_t].append(idx)

            for r_t in range(self.n_rxn_type):
                if r_t not in l.rxn_type:
                    self.excluded_rxn_type[r_t].append(idx)

        self.weight = [l.n / self.lg_cnt_sum for l in self.lg]

        # len_lgs = len(self.lg)
        # print(f"cnt\ttype\tcomb\tcnt\ttype\t total:{len_lgs}")
        # for i in range(n_rxn_cnt):
        #     for j in range(n_rxn_type):
        #         print(f"{i}\t{j}\t"
        #               f"{len_lgs - len(self.excluded_idx[i][j])}\t"
        #               f"{len_lgs - len(self.excluded_rc_cnt[i])}\t"
        #               f"{len_lgs - len(self.excluded_rxn_type[j])}")

    def __len__(self):
        return len(self.lg)

    def __getitem__(self, idx):
        return self.lg[idx]

    def get_excluded_idxs(self, center_cnt, rxn_type=None):
        if rxn_type is not None:
            return self.excluded_idx[center_cnt][rxn_type]
        else:
            return self.excluded_rc_cnt[center_cnt]
