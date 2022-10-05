import argparse
import os
import torch
import sys


parser = argparse.ArgumentParser()

# ===================== gpu id ===================== #
parser.add_argument('--gpu', type=int, default=-1)

# =================== random seed ================== #
parser.add_argument('--seed', type=int, default=1234)

# ==================== dataset ===================== #
parser.add_argument('--test_routes',
                    default='data/multi-step/retro_data/dataset/routes_possible_test_hard.pkl')
parser.add_argument('--starting_molecules', default='data/multi-step/retro_data/dataset/origin_dict.csv')

# ================== value dataset ================= #
parser.add_argument('--value_root', default='data/multi-step/retro_data/dataset')
parser.add_argument('--value_train', default='train_mol_fp_value_step')
parser.add_argument('--value_val', default='val_mol_fp_value_step')

# ================== one-step model ================ #
parser.add_argument('--mlp_model_dump',
                    default='tb_logs/retro1/version_495/checkpoints/epoch=239-step=236880.ckpt')
parser.add_argument('--mlp_templates',
                    default='data/multi-step/retro_data/one_step_model/template_rules_1.dat')

# ===================== all algs =================== #
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--expansion_topk', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--viz_dir', default='data/multi-step/viz')

# ===================== model ====================== #
parser.add_argument('--fp_dim', type=int, default=2048)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=128)

# ==================== training ==================== #
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_epoch_int', type=int, default=1)
parser.add_argument('--save_folder', default='saved_models')

# ==================== evaluation =================== #
parser.add_argument('--use_value_fn', action='store_true')
parser.add_argument('--value_model', default='best_epoch_final_4.pt')
parser.add_argument('--result_folder', default='results')

args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
