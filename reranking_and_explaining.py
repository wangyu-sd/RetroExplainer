# !/usr/bin/python3
# @File: train.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.27.19
import os
import argparse
import torch
from model.RetroAGT import RetroAGT
from data.datasets import RerankingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import one_hot
from model.util import bond_fea2type
import nums_from_string

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

BOND_TYPE_INT_TO_BOND_ORDER = {0: 0, 1: 1, 2: 1.5, 3: 2, 4: 3}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    # dataset configuration
    parser.add_argument('--dataset', type=str, default="GLN_200topk_200maxk_noGT_19260817_test")
    parser.add_argument('--model_path', type=str,
                        default='tb_logs/retro2/version_41/checkpoints/epoch=374-step=29250.ckpt')
    parser.add_argument('--not_fast_read', default=False, action='store_true')
    parser.add_argument('--lg_path', type=str,
                        default='/mnt/solid/wy/retro2/data/uspto50k_2/processed/leaving_group.pt')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--need_action', default=False, action='store_true')

    args = parser.parse_args()

    dataset = RerankingDataset(root="./data/reranking", dataset=args.dataset, lg_path=args.lg_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Building Model...")
    model = build_model(args).to(args.cuda)
    print("Finished Model...")

    print("Predicting...")
    model.eval()
    model.state = 'reranking'
    origin_top_k = TopKAccRecorder()
    model_top_k = TopKAccRecorder()
    model_correct_action = []
    model_best_action = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Epoch')):
        batch = data_to_device(batch, args.cuda)
        bsz = batch['origin_idx'].shape[0]

        pred_state = []
        # because the re-ranking task is aware of center cnt, we add rxn_cnt condition as prior knowledge
        for rxn_cnt in range(10):
            batch['center_cnt'] = torch.tensor([rxn_cnt] * bsz).to(args.cuda)
            rc_adj_prob, h_prob, lg_prob, rc_fea = model(batch)
            rc_adj_prob = rc_adj_prob.sigmoid()
            h_prob = h_prob.softmax(dim=-1)
            lg_prob = lg_prob.softmax(dim=-1)
            pred_state.append({
                'rc_adj_prob': rc_adj_prob,
                'h_prob': h_prob,
                'lg_prob': lg_prob,
                'rc_fea': rc_fea,
            })
        product_bond_order_adj = bond_fea2type(batch['product']['bond_adj'] - 1)

        with tqdm(total=bsz, leave=False) as pbar:
            pbar.set_description('Step')
            for idx_data in range(bsz):
                cur_product = []
                predictions = dataset.get_predictions(batch['origin_idx'][idx_data],
                                                      end_idx=50)
                origin_rank = batch['origin_rank'][idx_data]
                if predictions:
                    origin_top_k.count(origin_rank, n_pred=len(predictions))

                solutions = []
                n_pro = int(batch['product']['n_atom'][idx_data])

                for pred, rank in tqdm(predictions, desc='Top_K_Predictions', leave=False):
                    pred = data_to_device(pred, args.cuda)
                    center_cnt = int(pred['center_cnt'])
                    cur_state_dict = pred_state[center_cnt]
                    cur_rc_adj = cur_state_dict['rc_adj_prob'][idx_data, :n_pro, :n_pro]
                    cur_h = cur_state_dict['h_prob'][idx_data, :n_pro]

                    lg_dic = pred['lg']
                    n_lg = int(lg_dic['n_atom'])

                    shared_atom_fea_lg, masked_adj_lg = model.emb(lg_dic['atom_fea'].unsqueeze(0),
                                                                  lg_dic['bond_adj'].unsqueeze(0),
                                                                  lg_dic['dist_adj'].unsqueeze(0),
                                                                  torch.tensor([center_cnt], dtype=torch.long,
                                                                               device=lg_dic['atom_fea'].device),
                                                                  None,
                                                                  dist3d_adj=None,
                                                                  contrast=True)
                    shared_atom_fea_lg = model.shared_encoder(shared_atom_fea_lg, masked_adj_lg)
                    shared_atom_fea_lg[:, 1:] += model.gate_embedding(pred['gate_token'])

                    ct_fea_lg = model.ct_encoder(shared_atom_fea_lg, masked_adj_lg)[:, 1:]
                    # rc_fea = self.ct_encoder(shared_atom_fea, masked_adj)[:, 1:]

                    ct_prob_fea = model.ct_adj_fn(cur_state_dict['rc_fea'][[idx_data], 1:], ct_fea_lg, None)[:, :,
                                  :model.max_ct_atom]
                    ct_prob = model.ct_out_fn(ct_prob_fea).squeeze(-1) + \
                              torch.where(batch['product']['atom_fea'][[idx_data], 0] > 0, 0., -1e3)[:, :, None]

                    # Find LG
                    cur_lg_idx = int(pred['lg_id'])

                    if cur_lg_idx < 0:
                        lg_fea2 = model.lg_encoder(shared_atom_fea_lg, masked_adj_lg)[:, 0]
                        lg_prob2 = model.lg_out_fn(lg_fea2)
                        cur_lg_idx = lg_prob2.argmax(dim=-1).item()

                    cur_lg_score = cur_state_dict['lg_prob'][idx_data, cur_lg_idx]
                    cur_lg_score = -cur_lg_score.log()

                    ct_prob = ct_prob.squeeze(0).sigmoid()[:n_pro, :n_lg]

                    if args.need_action:
                        cur_action = [f"LGM|Select Leaving Group with Index {cur_lg_idx} and Energy %.2f" % cur_lg_score]
                    it_score = cur_lg_score - (1 - cur_rc_adj).log().sum() \
                               - cur_h[:, 3].log().sum() - (1 - ct_prob).log().sum()
                    if args.need_action:
                        cur_action += [f"IT|Initial Energy:%.2f" % it_score]

                    # Find LGC
                    real_ct = pred['ct_target'][:n_pro, :n_lg].bool()
                    ct_score = -ct_prob[real_ct].log().sum() + (1 - ct_prob[real_ct]).log().sum()
                    if args.need_action:
                        for pro_idx, lg_idx in real_ct.nonzero().tolist():
                            cur_cost = -ct_prob[pro_idx, lg_idx].log() + (1 - ct_prob[pro_idx, lg_idx]).log()
                            cur_bond_order = pred["rea_bond_adj"][pro_idx, lg_idx]
                            cur_bond_order = BOND_TYPE_INT_TO_BOND_ORDER[int(cur_bond_order)]
                            cur_action += [f"LGC|Add Bonds: between {pro_idx} and {lg_idx + n_pro} with Bond Type "
                                           f"{cur_bond_order} and Cost %.2f" % cur_cost]

                    # Find RC
                    real_rc = pred['rc_target'][:n_pro, :n_pro].bool()
                    rc_score = -cur_rc_adj[real_rc].log().sum() + (1 - cur_rc_adj[real_rc]).log().sum()
                    if args.need_action:
                        for pro_idx_1, pro_idx_2 in real_rc.nonzero().tolist():
                            if pro_idx_1 >= pro_idx_2:
                                continue
                            cur_cost = -cur_rc_adj[pro_idx_1, pro_idx_2].log() + (1 - cur_rc_adj[pro_idx_1, pro_idx_2]).log()
                            cur_cost += -cur_rc_adj[pro_idx_2, pro_idx_1].log() + (1 - cur_rc_adj[pro_idx_2, pro_idx_1]).log()

                            cur_bond_order = pred["rea_bond_adj"][pro_idx_1, pro_idx_2]
                            cur_bond_order = BOND_TYPE_INT_TO_BOND_ORDER[int(cur_bond_order)]

                            origin_bond_order = product_bond_order_adj[idx_data, pro_idx_1, pro_idx_2].item()
                            cur_action += [f"Replace Bonds: between {pro_idx_1} and {pro_idx_2}, "
                                           f"from Bond Type {origin_bond_order} to Bond Type"
                                           f"{cur_bond_order} with Cost %.2f" % cur_cost]

                    real_hc = pred['rc_h'][:n_pro]
                    mask_select = one_hot(real_hc, 7).bool()
                    hc_score = -cur_h[mask_select].log().sum() + (1 - cur_h[:, 3]).log().sum()
                    if args.need_action:
                        for pro_idx, n_hc in mask_select.nonzero().tolist():
                            cur_cost = -cur_h[pro_idx, n_hc].log() + (1 - cur_h[pro_idx, 3]).log()
                            cur_action += [f"H number change {n_hc - 3} cost of atom {pro_idx}: %.2f" % cur_cost]

                    total_score = it_score + ct_score + rc_score + hc_score

                    if not args.need_action:
                        cur_action = []
                    solutions.append((float(total_score), rank, rank == origin_rank, cur_action))

                solutions.sort(key=lambda x: x[0])
                model_rank = 9999
                if solutions:
                    model_best_action.append(solutions[0])
                    for rank_idx in range(len(solutions)):
                        if solutions[rank_idx][2]:
                            model_rank = rank_idx
                            model_correct_action.append(solutions[rank_idx])
                            break
                    if model_rank == 9999:
                        model_correct_action.append(([], 10000, False))

                    model_top_k.count(model_rank, n_pred=len(predictions))

                pbar.set_postfix({"origin_1_acc": origin_top_k.top_k_acc_dict[f'top_{1}_acc'],
                                  "model_1_acc": model_top_k.top_k_acc_dict[f'top_{1}_acc'],
                                  "origin_per": f"{origin_top_k.get_percentage_rank():.2f}",
                                  "model_per": f"{model_top_k.get_percentage_rank():.2f}"})
                pbar.update(1)

    origin_top_k.normalize(len(dataset))
    model_top_k.normalize(len(dataset))

    print(args)
    print("Origin Top K Acc:", origin_top_k.top_k_acc_dict)
    print("Model Top K Acc:", model_top_k.top_k_acc_dict)
    print("Origin_percentage", f"{origin_top_k.get_percentage_rank():.2f}")
    print("Model_percentage", f"{model_top_k.get_percentage_rank():.2f}")


def build_model(args):
    model = RetroAGT.load_from_checkpoint(
        args.model_path,
        lg_path=args.lg_path,
        strict=False,
    )
    return model


class TopKAccRecorder:
    def __init__(self):
        self.top_k_acc_dict = {f"top_{k}_acc": 0 for k in [1, 3, 5, 10, 50]}
        self.cur_k = []

    def count(self, rank, n_pred=None):
        if isinstance(rank, torch.Tensor):
            rank = rank.item()
        elif isinstance(rank, str):
            rank = int(rank)

        if n_pred is not None and rank < 50:
            self.cur_k.append((rank+1)/n_pred)
        else:
            self.cur_k.append(1)

        for k in self.top_k_acc_dict.keys():
            if rank < nums_from_string.get_nums(k)[0]:
                self.top_k_acc_dict[k] += 1

    def normalize(self, total):
        for k in self.top_k_acc_dict.keys():
            self.top_k_acc_dict[k] /= total

    def get_percentage_rank(self):
        return sum(self.cur_k) / len(self.cur_k)


def prob2energy(prob):
    return -prob.log()


def data_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device=device)
    elif isinstance(batch, tuple):
        batch = tuple(data_to_device(ele, device) for ele in batch)
    elif isinstance(batch, dict):
        batch = {k: data_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        batch = [data_to_device(ele, device) for ele in batch]
    return batch


if __name__ == '__main__':
    main()
