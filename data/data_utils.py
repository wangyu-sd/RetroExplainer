# !/usr/bin/python3
# @File: data_utils.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.20.22
import torch
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from functools import cmp_to_key
from rdkit.Chem import rdDistGeom as molDG
BOND_ORDER_MAP = {0: 0, 1: 1, 1.5: 2, 2: 3, 3: 4}

# def build_multi_hop_adj(adjs, n_hop, cuda=0):
#     """
#     :param adjs: [n_fea_type, n_atom, n_atom]
#     :param cuda:
#     :param n_hop:
#     :return: [n_hop, n_fea_type, n_atom, n_atom]
#     """
#     n_fea_type, n_atom, _ = adjs.size()
#     new_adjs = torch.zeros(n_hop, n_fea_type, n_atom, n_atom, device=cuda, dtype=torch.half)
#
#     new_adjs[0] = adjs
#     for hop in range(1, n_hop):
#         for fea_idx in range(n_fea_type):
#             new_adjs[hop, fea_idx] = torch.mm(new_adjs[hop - 1, fea_idx], new_adjs[0, fea_idx])
#     return new_adjs.detach()


def padding_mol_info(mol_dic, max_nodes, padding_idx=0):
    """
    :param mol_dic={
            atom_fea: [n_atom_fea_type, n_atom],
            bond_adj: [n_hop, n_adj_type, n_atom, n_atom],
            ...,
    }
    :param max_nodes:
    :param padding_idx: value of padding token
    :return: {
            atom_fea: [n_atom_fea_type, max_nodes],
            bond_adj: [n_hop, n_adj_type, max_nodes, max_nodes],
            ...,
    }
    """
    n_atom_fea, n_atom = mol_dic['atom_fea'].size()
    new_atom_fea = torch.zeros(n_atom_fea, max_nodes, dtype=mol_dic['atom_fea'].dtype)
    new_atom_fea[:, :n_atom] = mol_dic['atom_fea']
    mol_dic['atom_fea'] = new_atom_fea

    # n_hop, n_adj_type, _, _ = mol_dic['bond_adj'].size()
    # # Considering virtual node , we set max_nodes + 1   // deleted
    # new_bond_adj = torch.zeros(n_hop, n_adj_type, max_nodes, max_nodes, dtype=torch.uint8)
    # new_bond_adj[:, :, :n_atom, :n_atom] = mol_dic['bond_adj'] + 1
    # # Plussing 1 is to make difference between no edge and padding token

    # Considering virtual node , we set max_nodes + 1   // deleted
    new_bond_adj = torch.zeros(max_nodes, max_nodes, dtype=torch.uint8)
    new_bond_adj[:n_atom, :n_atom] = mol_dic['bond_adj'] + 1
    # Plussing 1 is to make difference between no edge and padding token

    mol_dic['bond_adj'] = new_bond_adj
    mol_dic['dist_adj'] = pad_adj(mol_dic['dist_adj'] + 1e-5, max_nodes)
    if 'dist_adj_3d' in mol_dic.keys() and mol_dic['dist_adj_3d'] is None:
        del mol_dic['dist_adj_3d']
    elif 'dist_adj_3d' in mol_dic.keys():
        mol_dic['dist_adj_3d'] = pad_adj(mol_dic['dist_adj_3d'] + 1e-5, max_nodes)


def smile_to_mol_info(smile, calc_dist=True, use_3d_info=False):
    mol = Chem.MolFromSmiles(smile)
    bond_adj = get_bond_adj(mol)
    dist_adj = get_dist_adj(mol) if calc_dist else None
    dist_adj_3d = get_dist_adj(mol, use_3d_info) if calc_dist else None
    atom_fea, n_atom = get_atoms_info(mol)
    return {
        "mol": mol,
        "bond_adj": bond_adj,
        "dist_adj": dist_adj,
        "dist_adj_3d": dist_adj_3d,
        "atom_fea": atom_fea,
        "n_atom": n_atom
    }


def get_atoms_info(mol):
    atoms = mol.GetAtoms()
    n_atom = len(atoms)
    atom_fea = torch.zeros(7, n_atom, dtype=torch.half)
    AllChem.ComputeGasteigerCharges(mol)
    for idx, atom in enumerate(atoms):
        atom_fea[0, idx] = atom.GetAtomicNum()
        atom_fea[1, idx] = atom.GetTotalDegree() + 1
        atom_fea[2, idx] = int(atom.GetHybridization()) + 1
        atom_fea[3, idx] = atom.GetTotalNumHs() + 1
        atom_fea[4, idx] = atom.GetIsAromatic() + 1
        for n_ring in range(3, 9):
            if atom.IsInRingSize(n_ring):
                atom_fea[5, idx] = n_ring + 1
                break
        else:
            if atom.IsInRing():
                atom_fea[5, idx] = 10
        atom_fea[6, idx] = atom.GetDoubleProp("_GasteigerCharge")

    atom_fea = torch.nan_to_num(atom_fea)
    return atom_fea, n_atom


def get_bond_order_adj(mol):
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.uint8)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_adj[i, j] = bond_adj[j, i] = BOND_ORDER_MAP[bond.GetBondTypeAsDouble()]
    return bond_adj


def get_bond_adj(mol):
    """
    :param mol: rdkit mol
    :return: multi graph for {
                sigmoid_bond_graph,
                pi_bond_graph,
                2pi_bond_graph,
                aromic_graph,
                conjugate_graph,
                ring_graph,
    }
    """
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.uint8)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        bond_adj[i, j] = bond_adj[j, i] = 1
        if bond_type in [2, 3]:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 1)
        if bond_type == 3:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 2)
        if bond_type == 1.5:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 3)
        if bond.GetIsConjugated():
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 4)
        if bond.IsInRing():
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 5)
    return bond_adj


def get_tgt_adj_order(product, reactants):
    atom_idx2map_idx = {}
    pro_map_ids = []
    for atom in product.GetAtoms():
        atom_idx2map_idx[atom.GetIdx()] = atom.GetAtomMapNum()
        pro_map_ids.append(atom.GetAtomMapNum())

    map_idx2atom_idx = {0: []}
    r_atoms = list(reactants.GetAtoms())
    for atom in r_atoms:
        cur_mp_id = atom.GetAtomMapNum()
        if atom.GetAtomMapNum() not in pro_map_ids:
            map_idx2atom_idx[0].append(atom.GetIdx())
        else:
            map_idx2atom_idx[atom.GetAtomMapNum()] = atom.GetIdx()

    order = []
    for atom in product.GetAtoms():
        order.append(map_idx2atom_idx[atom_idx2map_idx[atom.GetIdx()]])

    leaving_group = map_idx2atom_idx[0]
    bond_list = [bond for bond in reactants.GetBonds()
                 if bond.GetBeginAtomIdx() in leaving_group
                 or bond.GetEndAtomIdx() in leaving_group]

    def lg_atom_cmp(bond_1, bond_2):
        cmp_res = atom_cmp(r_atoms[bond_1[1]], r_atoms[bond_2[1]])
        if not cmp_res:
            return cmp_res
        return atom_cmp(r_atoms[bond_1[2]], r_atoms[bond_2[2]])

    gate_cnt, is_gate, bridge = [], True, []

    while bond_list:
        rm_bond, rm_bond_idx = [], []
        exit_reagent = True
        for bond_id, bond in enumerate(bond_list):
            aid1, aid2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if aid1 in leaving_group and aid2 in leaving_group:
                continue
            elif aid1 in leaving_group:
                aid1, aid2 = aid2, aid1
            rm_bond.append((bond, aid1, aid2))
            rm_bond_idx.append(bond_id)
            exit_reagent = False

        if exit_reagent:
            pass

        rm_bond.sort(key=cmp_to_key(lg_atom_cmp))

        if is_gate:
            gate_atoms = []
            for b in rm_bond:
                if b[2] not in gate_atoms:
                    gate_atoms.append(b[2])
                    gate_cnt.append(1)
                else:
                    gate_atom_idx = gate_atoms.index(b[2])
                    gate_cnt[gate_atom_idx] += 1
                bridge.append((b[1], float(b[0].GetBondTypeAsDouble())))
            is_gate = False

        bond_list = [
            bond_list[i]
            for i in range(len(bond_list))
            if i not in rm_bond_idx
        ]

        for rm_b in rm_bond:
            if rm_b[2] in leaving_group:
                order.append(rm_b[2])
                leaving_group.remove(rm_b[2])

    return torch.tensor(order, dtype=torch.long), gate_cnt, bridge


def atom_cmp(a1, a2):
    an1, an2 = a1.GetAtomicNum(), a2.GetAtomicNum()
    if an1 != an2:
        return an1 - an2
    hy1, hy2 = a1.GetHybridization(), a2.GetHybridization()
    return hy1 - hy2


def get_dist_adj(mol, use_3d_info=False):
    if use_3d_info:
        m2 = Chem.AddHs(mol)
        is_success = AllChem.EmbedMolecule(m2, enforceChirality=False)
        if is_success == -1:
            dist_adj = None
        else:
            AllChem.MMFFOptimizeMolecule(m2)
            m2 = Chem.RemoveHs(m2)
            dist_adj = (-1 * torch.tensor(AllChem.Get3DDistanceMatrix(m2), dtype=torch.float))
    else:
        dist_adj = (-1 * torch.tensor(molDG.GetMoleculeBoundsMatrix(mol), dtype=torch.float))

    return dist_adj


def pad_1d(x, n_max_nodes):
    if not isinstance(x, Tensor):
        raise TypeError(type(x), "is not a torch.Tensor.")
    n = x.size(0)
    new_x = torch.zeros(n_max_nodes).to(x)
    new_x[:n] = x
    return new_x


def pad_adj(x, n_max_nodes):
    if x is None:
        return None
    if not isinstance(x, Tensor):
        raise TypeError(type(x), "is not a torch.Tensor.")
    n = x.size(0)
    assert x.size(0) == x.size(1)
    new_x = torch.zeros([n_max_nodes, n_max_nodes], dtype=x.dtype)
    new_x[:n, :n] = x
    return new_x
