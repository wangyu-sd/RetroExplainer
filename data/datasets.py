import os
import pandas as pd
import os.path as osp
from torch_geometric.data import Dataset
from torch.utils import data
from tqdm import tqdm
import torch
from rxnmapper import BatchedMapper
from data.data_utils import smile_to_mol_info, get_tgt_adj_order, padding_mol_info, \
    get_bond_order_adj, pad_adj, pad_1d, get_tgt_adj_order_mit, shuffle_map_numbers, map_id_alignment, check_reactant
from data.LeavingGroup import LeavingGroup
import copy

CUT_OFF = 10


class RerankingDataset(Dataset):
    @property
    def raw_file_names(self):
        return [osp.join("rxnebm_data", self.dataset + ".csv")]

    @property
    def processed_file_names(self):
        return [f"product_{idx}.pt" for idx in range(self.size)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, osp.join("processed", self.dataset))

    def __init__(self, root, dataset, max_node=200, max_gate_num_size=4, min_node=3, use_3d_info=False, max_lg_na=23,
                 lg_path="/mnt/solid/wy/retro2/data/uspto50k_2/processed/leaving_group.pt", cuda=0):
        self.root = root
        self.dataset = dataset
        self.max_node = max_node
        self.max_gate_num_size = max_gate_num_size
        self.min_node = min_node
        self.use_3d_info = use_3d_info
        self.max_lg_na = max_lg_na
        self.lg_path = lg_path
        self.cuda = cuda
        self.size_path = osp.join(self.root, "processed", self.dataset, "size.pt")
        if osp.exists(self.size_path):
            self.size = torch.load(self.size_path)
        else:
            self.size = 0
        super().__init__(root)

    def process(self):
        cur_idx = 0
        max_rxn_cnt = 10
        leaving_group = torch.load(self.lg_path)

        rxn_mapper = BatchedMapper(batch_size=200)
        for raw_path in self.raw_paths:
            cur_csv = pd.read_csv(raw_path)
            cur_length = len(cur_csv)

            for idx in tqdm(range(cur_length), desc="Products"):
                reactant_smi, product_smi = cur_csv.iloc[idx]['orig_rxn_smi'].split(">>")
                product = smile_to_mol_info(product_smi)
                reactant = smile_to_mol_info(reactant_smi)

                prediction_list = []
                for index_precursors in range(200):
                    cur_reactant_smi = str(cur_csv.iloc[idx][f'cand_precursor_{index_precursors + 1}'])
                    if cur_reactant_smi != '9999':
                        prediction_list.append(product_smi + ">>" + cur_reactant_smi)
                    else:
                        break

                prediction_list = list(rxn_mapper.map_reactions(prediction_list))

                cur_index_prediction = 0
                for index_precursors in tqdm(range(len(prediction_list)), desc="Top_K_Predictions", leave=False):
                    if cur_csv['rank_of_true_precursor'].iloc[idx] != index_precursors:
                        if prediction_list[index_precursors] == ">>":
                            continue
                        cur_product_smi, cur_reactant_smi = prediction_list[index_precursors].split(">>")
                        cur_reactant_smi, cur_product_smi = \
                            map_id_alignment(cur_reactant_smi, cur_product_smi, product['mol'], return_smiles=True)
                        cur_reactant_smi = check_reactant(cur_reactant_smi)
                        cur_reactant = smile_to_mol_info(cur_reactant_smi)
                    else:
                        cur_reactant = copy.deepcopy(reactant)
                        cur_product_smi, cur_reactant_smi = product_smi, reactant_smi

                    cur_product = copy.deepcopy(product)

                    try:
                        order, gate_num, bridge = get_tgt_adj_order(cur_product['mol'], cur_reactant['mol'])
                    except (ValueError, KeyError) as e:
                        # print(e)
                        continue

                    cur_reactant['atom_fea'] = cur_reactant['atom_fea'][:, order]
                    cur_reactant['bond_adj'] = cur_reactant['bond_adj'][order, :][:, order]
                    cur_reactant['dist_adj'] = cur_reactant['dist_adj'][order, :][:, order]

                    n_pro, n_rea = cur_product['n_atom'], cur_reactant['n_atom']

                    pro_bond_adj = get_bond_order_adj(cur_product['mol'])
                    rea_bond_adj = get_bond_order_adj(cur_reactant['mol'])[order][:, order]
                    rc_target = torch.zeros_like(pro_bond_adj)
                    rc_target[:n_pro, :n_pro] = rea_bond_adj[:n_pro, :n_pro]

                    rc_target = (~torch.eq(rc_target, pro_bond_adj))
                    center = rc_target.nonzero()
                    center_cnt = center.size(0) // 2

                    if center_cnt >= CUT_OFF:
                        continue

                    center_sparse = torch.zeros(2, 10) - 1

                    n_lg = n_rea - n_pro
                    if n_lg >= self.max_lg_na:
                        continue

                    if len(gate_num) >= self.max_gate_num_size:
                        continue

                    lg_dict = {"atom_fea": cur_reactant['atom_fea'][:, n_pro:].clone(),
                               "bond_adj": cur_reactant['bond_adj'][n_pro:, n_pro:],
                               'dist_adj': cur_reactant['dist_adj'][n_pro:, n_pro:],
                               'n_atom': n_lg}
                    # lg_dict['bond_adj'] = build_multi_hop_adj(lg_dict['bond_adj'], n_hop=4)
                    # lg_dict['atom_fea'][-1] = 0

                    # print(n_rea, n_pro, len(regents_idx), n_lg, lg_dict['atom_fea'].size(1))
                    # print(n_lg, lg_dict['atom_fea'][:, :n_lg].size(1))
                    assert n_lg == lg_dict['atom_fea'][:, :n_lg].size(1)
                    cur_lg = LeavingGroup(na=n_lg,
                                          atom_fea=lg_dict['atom_fea'][:, :n_lg],
                                          bond_adj=lg_dict['bond_adj'][:n_lg, :n_lg] + 1,
                                          gate_num=gate_num,
                                          center_cnt=[center_cnt],
                                          rxn_type=[0],
                                          bridge=[bridge],
                                          dist_adj=lg_dict['dist_adj'][:n_lg, :n_lg],
                                          )

                    if cur_lg not in leaving_group:
                        lg_id = -1
                        # lg_id = len(leaving_group)
                        # leaving_group.append(cur_lg)
                    else:
                        lg_id = leaving_group.index(cur_lg)
                        leaving_group[lg_id].n += 1
                        if bridge not in leaving_group[lg_id].bridge:
                            leaving_group[lg_id].bridge.append(bridge)
                        if center_cnt not in leaving_group[lg_id].center_cnt:
                            leaving_group[lg_id].center_cnt.append(center_cnt)

                    # product['bond_adj'] = build_multi_hop_adj(product['bond_adj'], n_hop=4)
                    # padding
                    padding_mol_info(lg_dict, self.max_lg_na)
                    padding_mol_info(cur_product, self.max_node)
                    # rea_bond_adj = pad_adj(rea_bond_adj, self.max_node)
                    # rc_target = pad_adj(rc_target, self.max_node)
                    # rc_target_atom = pad_1d(rc_target_atom, self.max_node)

                    rc_h = torch.zeros(self.max_node) + 3
                    rc_h[:n_pro] = cur_reactant['atom_fea'][3, :n_pro] - cur_product['atom_fea'][3, :n_pro] + 3
                    if rc_h.max() > 6 or rc_h.min() < 0:
                        continue

                    gate_token = torch.zeros(self.max_lg_na, dtype=torch.long)
                    gate_token[:len(gate_num)] = torch.tensor(gate_num)

                    ct_target = torch.zeros(n_pro, self.max_gate_num_size, dtype=torch.long)
                    for i in range(len(gate_num)):
                        ct_target[:n_pro, i] = torch.where(rea_bond_adj[n_pro + i, :n_pro] > 0, 1, 0)

                    del cur_reactant
                    prediction = {
                        'lg': lg_dict,
                        'rc_h': rc_h.long(),
                        'rea_bond_adj': rea_bond_adj,
                        'rc_target': rc_target,
                        # 'center': center_sparse,
                        'ct_target': ct_target.float(),
                        'gate_token': gate_token.long(),
                        'center_cnt': center_cnt,
                        "lg_id": torch.LongTensor([lg_id]).squeeze(),
                    }

                    torch.save(prediction, osp.join(self.processed_dir,
                                                    f"product_{cur_idx}_p_{index_precursors}.pt"))
                    cur_index_prediction += 1

                padding_mol_info(product, self.max_node)
                del product['mol']
                rxn_data = {
                    "product": product,
                    "prediction_size": len(prediction_list),
                    "origin_rank": cur_csv['rank_of_true_precursor'].iloc[idx],
                    "origin_idx": idx,
                    "rxn_type": 0,
                }
                torch.save(rxn_data, osp.join(self.processed_dir, f"product_{cur_idx}.pt"))
                cur_idx += 1
            self.size = cur_idx
            torch.save(self.size, self.size_path)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.load(osp.join(self.processed_dir, f"product_{idx}.pt"))

    def get_predictions(self, idx, end_idx):
        return [(torch.load(osp.join(self.processed_dir, f"product_{idx}_p_{i}.pt")), i)
                for i in range(end_idx) if osp.exists(osp.join(self.processed_dir, f"product_{idx}_p_{i}.pt"))]


class MultiStepDataset(data.Dataset):
    def __init__(self, smiles, max_num_lg_atoms=70, max_num_atoms=100):
        super().__init__()
        self.raw_data = smiles
        self.max_lg_node = max_num_lg_atoms
        product = smile_to_mol_info(smiles)
        padding_mol_info(product, max_num_atoms)
        del product['mol']
        self.data = [{"product": product, 'rxn_type': 0, 'center_cnt': 0, }]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


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

    def __init__(self, root, data_split, fast_read=True, max_node=200, max_gate_num_size=10, max_regents_na=0,
                 min_node=3, use_3d_info=False, max_lg_na=50, save_cache=True, dataset_type='50k', known_regents=False):
        self.root = root
        self.lg_path = osp.join(root, osp.join('processed', 'leaving_group.pt'))
        # self.rxn_center_path = osp.join(root, osp.join('processed', 'rxn_center.pt'))
        self.min_node = min_node
        self.use_3d_info = use_3d_info
        self.data_split = data_split
        self.max_node = max_node
        self.max_gate_num_size = max_gate_num_size
        self.max_lg_na = max_lg_na
        self.max_regents_na = max_regents_na if known_regents else 0
        self.dataset_type = dataset_type
        self.known_regents = known_regents
        self.size_path = osp.join(osp.join(self.root, osp.join("processed", self.data_split)), "num_files.pt")
        if osp.exists(self.size_path):
            self.size = torch.load(self.size_path)
        else:
            self.size = 0
        self.need_reshuffle_mapped_atom = True if '50k' in self.dataset_type else False
        self.data = None
        super().__init__(root)
        if fast_read:
            data_cache_path = osp.join(self.root, osp.join('processed', f'cache_{data_split}.pt'))
            if osp.isfile(data_cache_path) and save_cache:
                print(f"read cache from {data_cache_path}...")
                self.data = torch.load(data_cache_path)
            else:
                self.fast_read = False
                self.data = [rxn_data for rxn_data in self]
                if save_cache:
                    torch.save(self.data, data_cache_path)

        self.fast_read = fast_read

    def process(self):
        cur_id = 0  # record the total id
        last_id = 0  # For recovery
        from collections import Counter
        count = Counter()
        try:
            leaving_group = torch.load(self.lg_path)
        except FileNotFoundError:
            leaving_group = []

        max_lg_na, max_atom_na, max_regents_na = 0, 0, 0
        os.makedirs(self.processed_dir, exist_ok=True)

        for raw_file_name in self.raw_file_names:
            # print(f"Processing the {raw_file_name} dataset to torch geometric format...\n")
            csv = pd.read_csv(osp.join(self.raw_dir, raw_file_name))
            reaction_list = csv['rxn_smiles']
            reactant_smarts_list = list(
                map(lambda x: x.split('>>')[0], reaction_list))
            product_smarts_list = list(
                map(lambda x: x.split('>>')[1], reaction_list))
            rxn_list = csv['class'] - 1 if 'class' in csv.keys() else None
            # with open(osp.join(self.processed_dir, "smiles_lists.csv"), "a+") as f:
            #     f.write("reactants, products\n")

            total = len(reactant_smarts_list)
            for idx, (reactant_smiles, product_smiles) \
                    in tqdm(enumerate(zip(reactant_smarts_list, product_smarts_list)), total=total):

                if idx < last_id:
                    count['idx < last_id'] += 1
                    continue
                if self.need_reshuffle_mapped_atom:
                    # some datasets like uspto50k need to shuffle the map numbers to avid the possible leakage
                    reactant_smiles, product_smiles = \
                        map_id_alignment(reactant_smiles, product_smiles, return_smiles=True)

                rxn_type = rxn_list[idx] if rxn_list is not None else 0
                product = smile_to_mol_info(product_smiles, use_3d_info=self.use_3d_info)
                if self.use_3d_info and product['dist_adj_3d'] is None:
                    count['self.use_3d_info'] += 1
                    continue
                elif product['n_atom'] <= self.min_node or product['n_atom'] >= self.max_node:
                    max_atom_na = max(max_atom_na, product['n_atom'])
                    count['product'] += 1
                    continue

                reactant = smile_to_mol_info(reactant_smiles, calc_dist=True, use_3d_info=False)

                regents_idx, regents = [], None
                try:
                    if self.dataset_type != 'mit':
                        if reactant['n_atom'] <= self.min_node or reactant['n_atom'] >= self.max_node:
                            count['reactant'] += 1
                            continue
                        order, gate_num, bridge = get_tgt_adj_order(product['mol'], reactant['mol'])
                    else:
                        order, gate_num, bridge, regents_idx = get_tgt_adj_order_mit(product['mol'], reactant['mol'],
                                                                                     reactant_smiles)
                        if reactant['n_atom'] <= self.min_node or reactant['n_atom'] >= self.max_node + len(
                                regents_idx):
                            count['reactant'] += 1
                            continue
                        if self.known_regents:
                            regents = {
                                'atom_fea': reactant['atom_fea'][:, regents_idx],
                                'bond_adj': reactant['bond_adj'][regents_idx, :][:, regents_idx],
                                'dist_adj': reactant['dist_adj'][regents_idx, :][:, regents_idx],
                            }
                            max_regents_na = max(max_regents_na, len(regents_idx))
                            if len(regents_idx) > self.max_regents_na:
                                count['regents'] += 1
                                continue

                except ValueError as e:
                    count['ValueError'] += 1
                    continue

                try:
                    bridge = tuple((int(reactant['atom_fea'][0, x[0]]), x[1]) for x in bridge)
                except IndexError as e:
                    print(bridge, e)
                    count['IndexError'] += 1
                    continue
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
                # rc_atoms = torch.zeros(self.max_node)

                if center_cnt > CUT_OFF:
                    count['center_cnt'] += 1
                    continue

                n_lg = n_rea - n_pro - len(regents_idx)
                if n_lg >= self.max_lg_na:
                    count['n_lg'] += 1
                    continue

                if len(gate_num) >= self.max_gate_num_size:
                    count['gate_num'] += 1
                    continue

                lg_dict = {"atom_fea": reactant['atom_fea'][:, n_pro:].clone(),
                           "bond_adj": reactant['bond_adj'][n_pro:, n_pro:],
                           'dist_adj': reactant['dist_adj'][n_pro:, n_pro:]}

                # lg_dict['bond_adj'] = build_multi_hop_adj(lg_dict['bond_adj'], n_hop=4)
                padding_mol_info(lg_dict, self.max_lg_na)
                # lg_dict['atom_fea'][-1] = 0

                # print(n_rea, n_pro, len(regents_idx), n_lg, lg_dict['atom_fea'].size(1))
                max_lg_na = max(max_lg_na, n_lg)
                assert n_lg == lg_dict['atom_fea'][:, :n_lg].size(1)
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
                padding_mol_info(product, self.max_node + self.max_regents_na)
                rea_bond_adj = pad_adj(rea_bond_adj, self.max_node + self.max_regents_na)
                rc_target = pad_adj(rc_target, self.max_node + self.max_regents_na)
                # rc_target_atom = pad_1d(rc_target_atom, self.max_node)

                rc_h = torch.zeros(self.max_node + self.max_regents_na) + 3
                rc_h[:n_pro] = reactant['atom_fea'][3, :n_pro] - product['atom_fea'][3, :n_pro] + 3
                if rc_h.max() > 6 or rc_h.min() < 0:
                    count['rc_h'] += 1
                    continue

                gate_token = torch.zeros(self.max_lg_na)
                gate_token[:len(gate_num)] = torch.tensor(gate_num)

                ct_target = torch.zeros(self.max_node + self.max_regents_na, self.max_gate_num_size, dtype=torch.long)
                for i in range(len(gate_num)):
                    ct_target[:n_pro, i] = torch.where(rea_bond_adj[n_pro + i, :n_pro] > 0, 1, 0)

                if regents is not None:
                    n_regents = len(regents_idx)
                    product['atom_fea'][:, n_pro:n_pro + n_regents] = regents['atom_fea']
                    product['bond_adj'][n_pro:n_pro + n_regents, n_pro:n_pro + n_regents] = regents['bond_adj']
                    product['dist_adj'][n_pro:n_pro + n_regents, n_pro:n_pro + n_regents] = regents['dist_adj']
                    product['n_atom'] += n_regents

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
                with open(osp.join(self.processed_dir, "smiles_lists.csv"), "a+") as f:
                    f.write(f"{reactant_smiles}, {product_smiles}\n")
                cur_id += 1

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
