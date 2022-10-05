# !/usr/bin/python3
# @File: dataset.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.18.21
import os
import pandas as pd
import os.path as osp
from torch_geometric.data import Dataset
from torch.utils import data
from tqdm import tqdm
import torch

from data.data_utils import smile_to_mol_info, get_tgt_adj_order, padding_mol_info, \
    get_bond_order_adj, pad_adj, pad_1d
from data.LeavingGroup import LeavingGroup 


class CacheDataset(data.Dataset):
    def __init__(self, root, data_split):
        super().__init__()
        self.root = root
        self.data_split = data_split
        self.data_path = osp.join(self.root, osp.join('processed', f'cache_{data_split}.pt'))
        self.data = torch.load(self.data_path)
        self.size = len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self, idx):
        return self.size


class RetroAGTDataSet(Dataset):
    @property
    def raw_file_names(self):
        return [self.data_split + ".csv"]

    @property
    def processed_file_names(self):
        return [f"rxn_data_{idx}.pt" for idx in range(self.size)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, osp.join("processed", self.data_split))

    def __init__(self, root, data_split, fast_read=True, max_node=50, max_gate_num_size=4,
                 min_node=3, use_3d_info=False, max_lg_na=30, save_cache=True):
        self.root = root
        self.lg_path = osp.join(root, osp.join('processed', 'leaving_group.pt'))
        # self.rxn_center_path = osp.join(root, osp.join('processed', 'rxn_center.pt'))
        self.min_node = min_node
        self.use_3d_info = use_3d_info
        self.data_split = data_split
        self.max_node = max_node
        self.max_gate_num_size = max_gate_num_size
        self.max_lg_na = max_lg_na
        self.size_path = osp.join(osp.join(self.root, osp.join("processed", self.data_split)), "num_files.pt")
        if osp.exists(self.size_path):
            self.size = torch.load(self.size_path)
        else:
            self.size = 0
        self.data = None
        super().__init__(root)
        if fast_read:
            data_cache_path = osp.join(self.root, osp.join('processed', f'cache_{data_split}.pt'))
            if osp.isfile(data_cache_path) and save_cache:
                print(f"read cache from {data_cache_path}...")
                self.data = torch.load(data_cache_path)
            else:
                self.fast_read = False
                self.data = [rxn_data for rxn_data in tqdm(self)]
                if save_cache:
                    torch.save(self.data, data_cache_path)

        self.fast_read = fast_read

        # print(self.data[0]['mask'].size())
        # for mol in ["product", "reactant"]:
        #     for k, v in self.data[0][mol].items():
        #         if isinstance(v, torch.Tensor):
        #             print(mol, k, v.size())
        #         else:
        #             print(mol, k, v)

    def process(self):
        cur_id = 0
        n_passed_mol = 0
        # cut_off = 15
        # self.max_gate_num_size = 4
        last_id = 0
        try:
            leaving_group = torch.load(self.lg_path)
        except FileNotFoundError:
            leaving_group = []

        os.makedirs(self.processed_dir, exist_ok=True)

        for raw_file_name in self.raw_file_names:
            print(f"Processing the {raw_file_name} dataset to torch geometric format...\n")
            csv = pd.read_csv(osp.join(self.raw_dir, raw_file_name))
            reaction_list = csv['rxn_smiles']
            reactant_smarts_list = list(
                map(lambda x: x.split('>>')[0], reaction_list))
            product_smarts_list = list(
                map(lambda x: x.split('>>')[1], reaction_list))
            rxn_list = csv['class'] - 1 if 'class' in csv.keys() else None

            total = len(reactant_smarts_list)
            for idx, (reactant_smiles, product_smiles) \
                    in tqdm(enumerate(zip(reactant_smarts_list, product_smarts_list)), total=total):

                if idx < last_id:
                    continue

                rxn_type = rxn_list[idx] if rxn_list is not None else 0
                product = smile_to_mol_info(product_smiles, use_3d_info=self.use_3d_info)
                if self.use_3d_info and product['dist_adj_3d'] is None:
                    n_passed_mol += 1
                    continue
                elif product['n_atom'] <= self.min_node or product['n_atom'] >= self.max_node:
                    n_passed_mol += 1
                    continue

                reactant = smile_to_mol_info(reactant_smiles, calc_dist=True, use_3d_info=False)

                if reactant['n_atom'] <= self.min_node or reactant['n_atom'] >= self.max_node:
                    n_passed_mol += 1
                    continue

                order, gate_num, bridge = get_tgt_adj_order(product['mol'], reactant['mol'])
                # reactant['dist_adj'] = reactant['dist_adj'][order][:, order]
                reactant['atom_fea'] = reactant['atom_fea'][:, order]
                reactant['bond_adj'] = reactant['bond_adj'][order, :][:, order]
                reactant['dist_adj'] = reactant['dist_adj'][order, :][:, order]

                n_pro, n_rea = product['n_atom'], reactant['n_atom']

                pro_bond_adj = get_bond_order_adj(product['mol'])
                rea_bond_adj = get_bond_order_adj(reactant['mol'])[order][:, order]
                rc_target = torch.zeros_like(pro_bond_adj)
                rc_target[:n_pro, :n_pro] = rea_bond_adj[:n_pro, :n_pro]

                rc_target = (~torch.eq(rc_target, pro_bond_adj))
                center = rc_target.nonzero()
                center_cnt = center.size(0) // 2
                # center = torch.stack([c for c in center if c[0] < c[1]]) if center_cnt > 0 else torch.zeros(2, 2) - 1
                rc_atoms = torch.zeros(self.max_node)

                # if center_cnt > cut_off:
                #     n_passed_mol += 1
                #     continue
                if n_rea - n_pro >= self.max_lg_na:
                    n_passed_mol += 1
                    continue
                try:
                    bridge = tuple((int(reactant['atom_fea'][0, x[0]]), x[1]) for x in bridge)
                except IndexError:
                    n_passed_mol += 1
                    continue

                if len(gate_num) >= self.max_gate_num_size:
                    n_passed_mol += 1
                    continue

                lg_dict = {"atom_fea": reactant['atom_fea'][:, n_pro:].clone(),
                           "bond_adj": reactant['bond_adj'][n_pro:, n_pro:],
                           'dist_adj': reactant['dist_adj'][n_pro:, n_pro:]}

                # lg_dict['bond_adj'] = build_multi_hop_adj(lg_dict['bond_adj'], n_hop=4)
                padding_mol_info(lg_dict, self.max_lg_na)
                # lg_dict['atom_fea'][-1] = 0
                n_lg = n_rea - n_pro
                cur_lg = LeavingGroup(na=n_lg,
                                      atom_fea=lg_dict['atom_fea'][:, :n_lg],
                                      bond_adj=lg_dict['bond_adj'][:n_lg, :n_lg],
                                      gate_num=gate_num,
                                      center_cnt=[center_cnt],
                                      rxn_type=[rxn_type],
                                      bridge=[bridge],
                                      dist_adj=lg_dict['dist_adj'][:n_lg, :n_lg],
                                      )

                # if raw_file_name not in ['train', 'valid'] and cur_lg not in leaving_group:
                #     n_passed_mol += 1
                #     continue

                if cur_lg not in leaving_group:
                    lg_id = len(leaving_group)
                    leaving_group.append(cur_lg)
                else:
                    lg_id = leaving_group.index(cur_lg)
                    leaving_group[lg_id].n += 1
                    if bridge not in leaving_group[lg_id].bridge:
                        leaving_group[lg_id].bridge.append(bridge)
                    if rxn_type is not None and rxn_type not in leaving_group[lg_id].rxn_type:
                        leaving_group[lg_id].rxn_type.append(rxn_type)
                    if center_cnt not in leaving_group[lg_id].center_cnt:
                        leaving_group[lg_id].center_cnt.append(center_cnt)

                # product['bond_adj'] = build_multi_hop_adj(product['bond_adj'], n_hop=4)
                # padding
                padding_mol_info(product, self.max_node)
                rea_bond_adj = pad_adj(rea_bond_adj, self.max_node)
                rc_target = pad_adj(rc_target, self.max_node)
                # rc_target_atom = pad_1d(rc_target_atom, self.max_node)


                rc_h = torch.zeros(self.max_node) + 3
                rc_h[:n_pro] = reactant['atom_fea'][3, :n_pro] - product['atom_fea'][3, :n_pro] + 3
                if rc_h.max() > 6 or rc_h.min() < 0:
                    n_passed_mol += 1
                    continue

                gate_token = torch.zeros(self.max_lg_na)
                gate_token[:len(gate_num)] = torch.tensor(gate_num)

                ct_target = torch.zeros(self.max_node, self.max_gate_num_size, dtype=torch.long)
                for i in range(len(gate_num)):
                    ct_target[:n_pro, i] = torch.where(rea_bond_adj[n_pro+i, :n_pro] > 0, 1, 0)

                del product['mol']
                del reactant
                rxn_data = {
                    "product": product,
                    'lg': lg_dict,
                    "rea_bond_adj": rea_bond_adj,
                    'rc_h': rc_h.long(),
                    'rc_target': rc_target.float(),
                    'rxn_type': rxn_type,
                    # 'center': center,
                    'ct_target': ct_target.float(),
                    'gate_token': gate_token.long(),
                    'center_cnt': center_cnt,
                    "lg_id": torch.LongTensor([lg_id]).squeeze()
                }

                torch.save(rxn_data, osp.join(self.processed_dir, f"rxn_data_{cur_id}.pt"))
                cur_id += 1

            print(f"Completed the {raw_file_name} dataset to torch geometric format...")

        # cnt = 0
        # print(f"|{self.data_split}: {len(leaving_group)}|", "*" * 89)
        # print("type\tnum\tatoms")
        # for lg in leaving_group:
        #     print(f"{lg.rxn_type}\t{lg.n}\t{lg.atoms}")
        #     cnt += lg.n
        print(f"|total={cur_id}|\t|passed={n_passed_mol}|", '*' * 90)
        self.size = cur_id
        torch.save(self.size, self.size_path)
        torch.save(leaving_group, self.lg_path)

    def len(self):
        return self.size

    def get(self, idx):
        if self.fast_read:
            rxn_data = self.data[idx]
        else:
            rxn_data = torch.load(osp.join(self.processed_dir, f"rxn_data_{idx}.pt"))
        return rxn_data
