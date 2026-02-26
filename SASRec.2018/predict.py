# 使用训练好的 SASRec 模型，以 train+valid 为序列对每个用户召回 top-K 物品，
# 结果映射回原始 viewer_id / mid 并保存为 {viewer_id: [mids]}。

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

from model import SASRec
from utils import data_partition


def _parse_value(s: str):
    s = s.strip()
    if s == 'True':
        return True
    if s == 'False':
        return False
    if s == 'None':
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load_args_from_run_dir(run_dir: str) -> dict:
    """从 run 目录的 args.txt 解析训练参数（每行 key,value）。"""
    path = os.path.join(run_dir, 'args.txt')
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line or '=' in line and line.startswith('run_id'):
                continue
            if ',' not in line:
                continue
            key, val = line.split(',', 1)
            key = key.strip()
            val = val.strip()
            out[key] = _parse_value(val)
    return out


def build_args_for_model(parsed: dict, dataset: str, device: str, top_k: int) -> argparse.Namespace:
    """用解析出的参数字典 + 命令行覆盖，构造模型所需的 args。"""
    defaults = {
        'dataset': dataset,
        'train_dir': '',
        'batch_size': 2048,
        'lr': 0.001,
        'maxlen': 50,
        'hidden_units': 32,
        'num_blocks': 2,
        'num_epochs': 40,
        'num_heads': 1,
        'dropout_rate': 0.2,
        'l2_emb': 0.0,
        'device': device,
        'inference_only': True,
        'state_dict_path': None,
        'norm_first': False,
        'log_every': 50,
        'tb_logdir': None,
        'topk': 10,
        'eval_recall_ann': False,
        'ann_top_M': top_k,
        'ann_use_faiss': False,
        'seed': 42,
    }
    for k, v in parsed.items():
        if k in defaults or k in ('dataset', 'device', 'maxlen', 'hidden_units', 'num_blocks', 'num_heads', 'dropout_rate', 'l2_emb', 'norm_first'):
            defaults[k] = v
    defaults['device'] = device
    return argparse.Namespace(**defaults)


def main():
    parser = argparse.ArgumentParser(description='SASRec 召回：train+valid 序列，每用户 top-K，结果映射回 viewer_id/mid')
    parser.add_argument('--dataset', required=True, help='与 train 一致的数据集名，对应 data/<dataset>.txt 与 mapping')
    parser.add_argument('--state_dict_path', required=True, help='模型权重路径（同目录下 args.txt 用于解析超参）')
    parser.add_argument('--output_dir', default='output', type=str, help='输出目录，其下保存 recall_result.json 与 recall_result.pkl')
    parser.add_argument('--top_k', default=100, type=int, help='每用户召回数量')
    parser.add_argument('--device', default=None, type=str, help='cuda / cpu，默认与 args.txt 或 cuda')
    parser.add_argument('--batch_size', default=512, type=int, help='召回时的用户 batch 大小，显存够可调大以加速')
    parser.add_argument('--data_dir', default='data', type=str, help='数据与 mapping 所在目录')
    args_cli = parser.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args_cli.state_dict_path))
    parsed = load_args_from_run_dir(run_dir)
    device = args_cli.device or parsed.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    args = build_args_for_model(parsed, args_cli.dataset, device, args_cli.top_k)

    # 数据与映射
    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = dataset
    mapping_path = os.path.join(args_cli.data_dir, f'{args_cli.dataset}_mapping.pkl')
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f'Mapping not found: {mapping_path}')
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    id2user = mapping['id2user']
    id2item = mapping['id2item']

    # 参与召回的用户：train+valid 至少 1 条
    maxlen = getattr(args, 'maxlen', 50)
    user_list = [u for u in range(1, usernum + 1) if len(user_train[u]) + len(user_valid[u]) >= 1]
    if not user_list:
        print('No users with train+valid sequence.')
        sys.exit(0)

    # 构造 (user_id, seq, rated) 列表，序列与 utils 一致：train+valid 右对齐
    seqs_list = []
    rated_list = []
    for u in user_list:
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        if user_valid[u]:
            seq[idx] = user_valid[u][0]
            idx -= 1
        for i in reversed(user_train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(user_train[u] + user_valid[u])
        rated.add(0)
        seqs_list.append(seq)
        rated_list.append(rated)

    # 模型
    model = SASRec(usernum, itemnum, args).to(args.device)
    state = torch.load(args_cli.state_dict_path, map_location=torch.device(args.device))
    model.load_state_dict(state)
    model.eval()

    top_k = min(args_cli.top_k, itemnum)
    batch_size = args_cli.batch_size
    item_emb = model.item_emb.weight.data[1:itemnum + 1].to(args.device)
    result_internal = {}  # u -> [item_id, ...]

    with torch.no_grad():
        for start in tqdm(range(0, len(seqs_list), batch_size), desc='recall'):
            end = min(start + batch_size, len(seqs_list))
            batch_seqs = np.asarray(seqs_list[start:end], dtype=np.int64)
            batch_seqs_t = torch.LongTensor(batch_seqs).to(args.device)
            user_vecs = model.get_user_vector(batch_seqs_t)
            scores = user_vecs @ item_emb.T
            # 先整批 mask 已交互，再只做一次 topk（避免在循环内重复做 topk）
            for b in range(end - start):
                rated_ids = rated_list[start + b]
                if rated_ids:
                    idx = torch.tensor([i - 1 for i in rated_ids], device=args.device, dtype=torch.long)
                    scores[b, idx] = float('-inf')
            _, topk_idx = scores.topk(top_k, dim=1)
            for b in range(end - start):
                u = user_list[start + b]
                retrieved = (topk_idx[b].cpu().numpy() + 1).tolist()
                result_internal[u] = retrieved

    # 映射回原始 viewer_id 与 mid
    result = {}
    for u in user_list:
        viewer_id = id2user[u]
        mids = [id2item[i] for i in result_internal[u] if i in id2item]
        result[viewer_id] = mids

    import json
    out_dir = args_cli.output_dir
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'recall_result.pkl')
    json_path = os.path.join(out_dir, 'recall_result.json')
    with open(pkl_path, 'wb') as f:
        pickle.dump(result, f)
    out_json = {str(k): list(v) for k, v in result.items()}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=0)
    print(f'Saved {len(result)} users to {out_dir}: recall_result.pkl, recall_result.json')


if __name__ == '__main__':
    main()
