# !/usr/bin/python
# @File: Connecter.py
# --coding:utf-8--
# @Author:Yu Wang
# @Email:as1003208735@foxmail.com
# @Time: 2022/4/23 16:19
# @Software: PyCharm
from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')


import torch
import logging
import time
from retro_star.retro_star.common import prepare_starting_molecules, prepare_mlp, \
    prepare_molstar_planner, smiles_to_fp
from retro_star.retro_star.model import ValueMLP
from retro_star.retro_star.utils import setup_logger
from model import prepare_single_step_model
import os

class RSPlanner:
    def __init__(self,
                 gpu=-1,
                 expansion_topk=50,
                 iterations=500,
                 use_value_fn=True,
                 starting_molecules="data/multi-step/retro_data/dataset/origin_dict.csv",
                 mlp_templates="data/multi-step/retro_data/one_step_model/template_rules_1.dat",
                 model_dump="tb_logs/retro1/version_495/checkpoints/epoch=239-step=236880.ckpt",
                 save_folder="data/multi-step/retro_data/saved_models",
                 value_model="best_epoch_final_4.pt",
                 fp_dim=2048,
                 viz=False,
                 viz_dir='viz'):

        setup_logger()
        device = torch.device('cuda:%d' % gpu if gpu >= 0 else 'cpu')
        starting_mols = prepare_starting_molecules(starting_molecules)

        one_step = prepare_single_step_model(model_dump)

        if use_value_fn:
            model = ValueMLP(
                n_layers=1,
                fp_dim=fp_dim,
                latent_dim=128,
                dropout_rate=0.1,
                device=device
            ).to(device)
            model_f = '%s/%s' % (save_folder, value_model)
            logging.info('Loading value nn from %s' % model_f)
            model.load_state_dict(torch.load(model_f, map_location=device))
            model.eval()

            def value_fn(mol):
                fp = smiles_to_fp(mol, fp_dim=fp_dim).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
        else:
            value_fn = lambda x: 0.

        self.plan_handle = prepare_molstar_planner(
            one_step=one_step,
            value_fn=value_fn,
            # starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
        )
        self.starting_mols = starting_mols

    def plan(self, target_mol, need_action=False):
        t0 = time.time()
        flag_removed = False
        if target_mol in self.starting_mols:
            flag_removed = True
            self.starting_mols.remove(target_mol)
        succ, msg = self.plan_handle(target_mol, self.starting_mols)

        if flag_removed:
            self.starting_mols.add(target_mol)
        if succ:
            result = {
                'succ': succ,
                'time': time.time() - t0,
                'iter': msg[1],
                'routes': msg[0].serialize(need_action=need_action),
                'route_cost': msg[0].total_cost,
                'route_len': msg[0].length
            }
            return result

        else:
            logging.info('Synthesis path for %s not found. Please try increasing '
                         'the number of iterations.' % target_mol)
            return None


if __name__ == '__main__':
    planner = RSPlanner(
        gpu=-1,
        use_value_fn=False,
        iterations=500,
        expansion_topk=10,
        viz=True,
        viz_dir='data/multi-step/viz',
        model_dump='model_saved/model_for_multi_step.ckpt'
    )

    # result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
    # print(result)
    #
    # smi_mapped = "[CH3:47][S:48](=[O:49])(=[O:50])[NH:54][CH2:57][CH2:58][CH2:2][c:35]1[cH:36][cH:37][c:15]2[c:16]([cH:34]1)[N:17]([CH2:20][c:21]1[cH:22][cH:23][c:24]([C:27](=[O:28])[N:29]3[CH2:30][CH:31]=[CH:32][CH2:33]3)[cH:25][cH:26]1)[C:18](=[O:19])[CH2:12][N:13]([C:38](=[O:46])[c:39]1[cH:40][cH:41][c:42]([Cl:45])[cH:43][cH:44]1)[CH2:14]2"
    # mol_mapped = Chem.MolFromSmiles(smi_mapped)
    # for atom in mol_mapped.GetAtoms():
    #     atom.SetAtomMapNum(0)
    smi_list = [
        "CC(CC1=CC=C2OCOC2=C1)NCC(O)C1=CC=C(O)C(O)=C1"
    ]
    for smi in smi_list:
        result = planner.plan(smi, need_action=True)
        print(result)


