# !/usr/bin/python
# @File: util.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022/5/10 21:58
# @Software: PyCharm
import torch
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from model.RetroAGT import RetroAGT
import re
bond_decoder_m = {1: Chem.BondType.SINGLE,
                  1.5: Chem.BondType.AROMATIC,
                  2: Chem.BondType.DOUBLE,
                  3: Chem.BondType.TRIPLE}
ATOM_VALENCY = {6:4, 7:3, 8:2, 9:1, 16:2, 17:1, 35:1}

def bond_fea2type(bond):
    bond = bond.long()
    bond_type = torch.where(bond > 1, 1., 0.).double()
    triple = torch.where(bond > 0, ((bond - 1) >> 2) % 2, 0)
    double = torch.where((bond > 0) * (triple < 1), ((bond - 1) >> 1) % 2, 0)
    aroma = torch.where(bond > 0, ((bond - 1) >> 3) % 2, 0)

    bond_type = torch.where(triple == 1, 3., bond_type)
    bond_type = torch.where(double == 1, 2., bond_type)
    bond_type = torch.where(aroma == 1, 1.5, bond_type)

    return bond_type


def correct_charge(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ['N']:
            cur_order = 0
            for bond in atom.GetBonds():
                cur_order += bond.GetBondTypeAsDouble()
            if cur_order == 4:
                atom.SetFormalCharge(1)
    return mol


def construct_mol(atoms, adj, adj_prob=None, atom_h_prob=None, need_map=True, threshold=0.5,
                  bond_decoder=bond_decoder_m, heigh_light=False):
    mol = Chem.RWMol()
    if heigh_light:
        hl_atom, hl_bond = [], []
    for idx, atom in enumerate(atoms):
        if atom != 0:
            atm = Chem.Atom(int(atom))
            if need_map:
                atm.SetAtomMapNum(int(idx) + 1)
            if atom_h_prob is not None and atom_h_prob[idx] != 0:
                atm.SetProp('atomNote', '%+dH' % int(atom_h_prob[idx]))
            mol.AddAtom(atm)

    for start, end in torch.nonzero(adj):
        if start < end:
            mol.AddBond(int(start), int(end), bond_decoder[float(adj[start, end])])

    if adj_prob is not None:
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if adj_prob[i, j].sigmoid() > threshold:
                bond.SetProp('bondNote', '\t%.2f' % adj_prob[i, j].sigmoid())

    if heigh_light:
        for idx, atom in enumerate(mol.GetAtoms()):
            if atom.HasProp('atomNote'):
                hl_atom.append(idx)
        for idx, atom in enumerate(mol.GetBonds()):
            if atom.HasProp('bondNote'):
                hl_bond.append(idx)
        return mol, hl_atom, hl_bond
    mol = correct_charge(mol)
    Chem.SanitizeMol(mol)
    return mol


def get_reaction(idx, batch, need_map=False):
    atoms = batch['product']['atom_fea'][idx, 0]
    bond_adj = bond_fea2type(batch['product']['bond_adj'][idx])
    # bond_adj = dataset[idx]['product']['bond_adj']
    product = construct_mol(atoms, bond_adj, adj_prob=None, need_map=need_map)
    #     print('product')
    #     Draw.MolToImage(product, size=(size,size))
    # rdChemReactions.ReactionFromMolecule(product)
    reactant = construct_reatacnt(batch['product'][idx], lgs[batch['lg_id'][idx]], batch['rea_bond_adj'][idx],
                                  need_map=need_map)
    return product, reactant


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def prepare_single_step_model(ckpt):
    # logging.info('Loading trained model from %s' % ckpt)
    single_step_model = RetroAGT.load_from_checkpoint(ckpt, strict=True)
    single_step_model.eval()
    single_step_model.state = 'inference'
    return single_step_model