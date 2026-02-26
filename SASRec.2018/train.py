import os
import sys
import time
import random
import argparse
import hashlib
import subprocess
from collections import deque

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=2048, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=32, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--log_every', default=50, type=int)
parser.add_argument('--tb_logdir', default=None, type=str)
parser.add_argument('--topk', default=10, type=int)
parser.add_argument('--eval_recall_ann', default=False, type=str2bool, help='是否用 ANN 评估召回 Recall')
parser.add_argument('--ann_top_M', default=100, type=int, help='ANN 每用户检索数量，用于召回评估')
parser.add_argument('--ann_use_faiss', default=False, type=str2bool, help='ANN 召回用 FAISS（否则用 PyTorch GPU）；CPU 时可设为 True')
parser.add_argument('--seed', default=42, type=int, help='随机种子，不设则每次不同，设则便于复现')

args = parser.parse_args()

# 实验根目录（同一 dataset+train_dir 下可有多条 run）
run_dir = os.path.join('logs', args.dataset + '_' + args.train_dir)
os.makedirs(run_dir, exist_ok=True)

# 本次运行的唯一 ID：时间戳 + 参数短哈希，便于 TensorBoard 多 run 对比与追溯
args_str = '\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])])
run_id = time.strftime('%Y%m%d_%H%M%S') + '_' + hashlib.sha256(args_str.encode()).hexdigest()[:8]
this_run_dir = os.path.join(run_dir, run_id)
os.makedirs(this_run_dir, exist_ok=True)

# 保存本次运行的全部参数
with open(os.path.join(this_run_dir, 'args.txt'), 'w') as f:
    f.write(args_str)
    f.write('\nrun_id,%s\n' % run_id)

def _get_git_info():
    try:
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)), stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--short'], cwd=os.path.dirname(os.path.abspath(__file__)), stderr=subprocess.DEVNULL, text=True).strip()
        return rev, 'dirty' if dirty else 'clean'
    except Exception:
        return None, None


if __name__ == '__main__':

    # 随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 保存复现信息：git、环境、命令行
    git_rev, git_dirty = _get_git_info()
    reproduce_lines = [
        'run_id=%s' % run_id,
        'seed=%d' % seed,
        'python=%s' % sys.version.split()[0],
        'torch=%s' % torch.__version__,
    ]
    if git_rev:
        reproduce_lines.append('git_rev=%s' % git_rev)
        reproduce_lines.append('git_dirty=%s' % git_dirty)
    reproduce_lines.append('cmd=%s' % ' '.join(sys.argv))
    with open(os.path.join(this_run_dir, 'reproduce.txt'), 'w') as f:
        f.write('\n'.join(reproduce_lines))
    print('run_id: %s  seed: %d  log_dir: %s' % (run_id, seed, this_run_dir))

    # ============================= 数据处理 =============================

    # 构建用户-物品索引
    u2i_index, i2u_index = build_index(args.dataset)
    print('user num: %d, item num: %d' % (len(u2i_index), len(i2u_index)))
    
    # 划分数据集
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    
    # 计算平均序列长度
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    # ============================= 日志记录 =============================
    
    # 记录日志（写入本次 run 目录，便于按 run 追溯）
    f = open(os.path.join(this_run_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr, val_recall) (test_ndcg, test_hr, test_recall) topk=%d\n' % args.topk)

    # TensorBoard：每条 run 单独子目录，tensorboard --logdir=logs/dataset_train_dir 可对比多 run
    tb_logdir = args.tb_logdir or os.path.join(this_run_dir, 'runs')
    writer = SummaryWriter(log_dir=tb_logdir)

    # ============================= 模型构建 =============================
    
    # 实例化数据采样器和模型
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    # 模型参数初始化： Xavier初始化
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    # 模型参数初始化：将padding idx的embedding设为0
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    # =============================================== Training loop ===============================================
    
    model.train() # enable model training

    epoch_start_idx = 1

    # 如果提供了预训练模型路径，则加载模型参数
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    # 仅推理模式
    if args.inference_only:
        model.eval()
        t_test = evaluate_fast(model, dataset, args)
        print('test (NDCG@%d: %.4f, HR@%d: %.4f)' % (args.topk, t_test[0], args.topk, t_test[1]))
        if args.eval_recall_ann:
            recall_ann = evaluate_recall_ann(model, dataset, args, top_M=args.ann_top_M)
            print('test Recall(ANN@%d): %.4f' % (args.ann_top_M, recall_ann))
    
    # 损失函数和优化器
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        epoch_t0 = time.time()
        losses = deque(maxlen=200)
        progress = tqdm(range(num_batch), total=num_batch, ncols=100, leave=False, unit='b', desc=f'Epoch {epoch}/{args.num_epochs}')
        
        # 训练一个epoch
        for step in progress:
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            # torch.norm(param) returns the square root of the sum of squared weights (‖w‖₂), 
            # should be torch.norm(param)**2 or the way below which is faster.
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.sum(param ** 2)    
            loss.backward()
            adam_optimizer.step()
            losses.append(loss.item())
            writer.add_scalar('train/loss_step', loss.item(), (epoch - 1) * num_batch + step)
            if (step + 1) % args.log_every == 0 or (step + 1) == num_batch:
                elapsed = time.time() - epoch_t0
                steps_done = step + 1
                avg_loss = sum(losses) / max(1, len(losses))
                eta = (elapsed / steps_done) * (num_batch - steps_done)
                msg = (
                    f"epoch {epoch}/{args.num_epochs} "
                    f"step {steps_done}/{num_batch} "
                    f"loss(avg{len(losses)}): {avg_loss:.4f} "
                    f"elapsed {elapsed:.1f}s eta {eta:.1f}s"
                )
                # print(msg)
                f.write(msg + "\n")
                f.flush()
                progress.set_postfix_str(f"loss {avg_loss:.4f} eta {eta:.1f}s")
        writer.add_scalar('train/loss_epoch', sum(losses) / max(1, len(losses)), epoch)

        # ============================= 模型评估 =============================

        # 每X个epoch评估一次模型
        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            # t_test = evaluate(model, dataset, args)
            t_test = evaluate_fast(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            msg = ('epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f)'
                    % (epoch, T, args.topk, t_valid[0], args.topk, t_valid[1], args.topk, t_test[0], args.topk, t_test[1]))
            if args.eval_recall_ann:
                print('', end='')
                recall_ann = evaluate_recall_ann(model, dataset, args, top_M=args.ann_top_M)
                msg += ', test Recall(ANN@%d): %.4f' % (args.ann_top_M, recall_ann)
                writer.add_scalar('eval/test_recall_ann', recall_ann, epoch)
            print(msg)
            writer.add_scalar('eval/valid_ndcg', t_valid[0], epoch)
            writer.add_scalar('eval/valid_hr', t_valid[1], epoch)
            writer.add_scalar('eval/test_ndcg', t_test[0], epoch)
            writer.add_scalar('eval/test_hr', t_test[1], epoch)


            # 当验证集 NDCG 或 HR 达到最佳时，保存模型
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                folder = this_run_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

            log_line = str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test)
            if args.eval_recall_ann:
                log_line += ' recall_ann@%d=%.4f' % (args.ann_top_M, recall_ann)
            f.write(log_line + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        # 保存最后一个epoch的模型参数
        if epoch == args.num_epochs:
            folder = this_run_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    writer.close()
    sampler.close()
    print("Done")
