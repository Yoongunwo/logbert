"""
logbert_seq.py — Sequential-ratio variant of logbert.py

Difference from logbert.py
---------------------------
The original logbert.py uses random sampling when train_ratio < 1 (via
sklearn's train_test_split inside generate_train_valid).

This file patches generate_train_valid so that train_ratio is applied
*sequentially from the start of the file* (time order):

  first  floor(N * train_ratio * (1 - valid_ratio))  sessions → train
  next   floor(N * train_ratio *       valid_ratio)   sessions → valid
  rest                                                sessions → ignored

Prerequisites
-------------
data_process.py must be run first to produce the 'train', 'test_normal',
and 'test_abnormal' files under ../output/bgl/.

For true time-sequential ordering, the data files should NOT be shuffled.
If you already ran data_process.py (which shuffles normal data), re-run it
after removing the line:
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)

Usage
-----
  python logbert_seq.py train
  python logbert_seq.py predict
  python logbert_seq.py vocab
"""

import sys
sys.path.append("../")

import argparse
import numpy as np
from tqdm import tqdm

from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from bert_pytorch.dataset.sample import fixed_window
from logdeep.tools.utils import *

# ── Sequential generate_train_valid ──────────────────────────────────────────
# train_log.py binds generate_train_valid in its own namespace via
#   "from bert_pytorch.dataset.sample import generate_train_valid"
# We patch that binding so Trainer.train() picks up the sequential version.

import bert_pytorch.train_log as _train_log_module

def _generate_train_valid_seq(data_path, window_size=20, adaptive_window=True,
                               sample_ratio=1, valid_size=0.1, output_path=None,
                               scale=None, scale_path=None, seq_len=None, min_len=0):
    """
    Sequential variant of generate_train_valid.

    Takes the FIRST (sample_ratio * total) sessions from the file in order,
    then splits that block sequentially into train / valid:
      train : first (1 - valid_size) fraction of the sampled block
      valid : remaining valid_size fraction of the sampled block
    No random shuffling is performed.
    """
    with open(data_path, 'r') as f:
        data_iter = f.readlines()

    total        = len(data_iter)
    num_session  = int(total * sample_ratio)           # how many to use in total
    test_size    = max(1, int(num_session * valid_size))
    train_size   = num_session - test_size

    print("before filtering short session")
    print(f"total sessions  : {total}")
    print(f"used  sessions  : {num_session}  (train_ratio={sample_ratio})")
    print(f"train size      : {train_size}")
    print(f"valid size      : {test_size}")
    print("=" * 40)

    logkey_trainset, time_trainset = [], []
    logkey_validset, time_validset = [], []

    for session_idx, line in enumerate(tqdm(data_iter, desc="Loading sessions")):
        if session_idx >= num_session:
            break
        logkeys, times = fixed_window(line, window_size, adaptive_window, seq_len, min_len)
        if session_idx < train_size:
            logkey_trainset += logkeys
            time_trainset   += times
        else:
            logkey_validset += logkeys
            time_validset   += times

    logkey_trainset = np.array(logkey_trainset, dtype=object)
    time_trainset   = np.array(time_trainset,   dtype=object)
    logkey_validset = np.array(logkey_validset, dtype=object)
    time_validset   = np.array(time_validset,   dtype=object)

    # Sort by descending sequence length (same as original, for efficient batching)
    if len(logkey_trainset) > 0:
        idx = np.argsort(-np.array(list(map(len, logkey_trainset))))
        logkey_trainset = logkey_trainset[idx]
        time_trainset   = time_trainset[idx]

    if len(logkey_validset) > 0:
        idx = np.argsort(-np.array(list(map(len, logkey_validset))))
        logkey_validset = logkey_validset[idx]
        time_validset   = time_validset[idx]

    print("=" * 40)
    print(f"Num of train seqs : {len(logkey_trainset)}")
    print(f"Num of valid seqs : {len(logkey_validset)}")
    print("=" * 40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset


# Patch the reference that Trainer.train() will look up at call time
_train_log_module.generate_train_valid = _generate_train_valid_seq

# ── Options (same as logbert.py) ──────────────────────────────────────────────

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

options["output_dir"] = "../output/bgl/"
options["model_dir"]  = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"
options["train_vocab"] = options['output_dir'] + 'train'
options["vocab_path"] = options["output_dir"] + "vocab.pkl"

options["window_size"]    = 128
options["adaptive_window"] = True
options["seq_len"]        = 512
options["max_len"]        = 512
options["min_len"]        = 10

options["mask_ratio"] = 0.5

options["train_ratio"] = 1      # ← set to e.g. 0.3 to train on first 30% (sequential)
options["valid_ratio"] = 0.1
options["test_ratio"]  = 1

options["is_logkey"] = True
options["is_time"]   = False

options["hypersphere_loss"]      = True
options["hypersphere_loss_test"] = False

options["scale"]      = None
options["scale_path"] = options["model_dir"] + "scale.pkl"

options["hidden"]     = 256
options["layers"]     = 4
options["attn_heads"] = 4

options["epochs"]        = 200
options["n_epochs_stop"] = 10
options["batch_size"]    = 32

options["corpus_lines"] = None
options["on_memory"]    = True
options["num_workers"]  = 5
options["lr"]               = 1e-3
options["adam_beta1"]       = 0.9
options["adam_beta2"]       = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]    = True
options["cuda_devices"] = None
options["log_freq"]     = None

options["num_candidates"] = 15
options["gaussian_mean"]  = 0
options["gaussian_std"]   = 1

seed_everything(seed=1234)
print("device", options["device"])
print("features logkey:{} time:{}".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])
print("[logbert_seq] generate_train_valid → sequential sampling (no random shuffle)")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    train_parser.add_argument("-r", "--ratio", type=float, default=None,
                              help="Override train_ratio (0~1]. "
                                   "First ratio*total sessions used, time-sequential.")
    train_parser.add_argument("--model_dir", type=str, default=None,
                              help="Override model output directory (default: ../output/bgl/bert/)")

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std",  type=float, default=1)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.set_defaults(mode='eval')
    eval_parser.add_argument("-r", "--ratio", type=float, default=None,
                             help="Same ratio used during training. "
                                  "Last (1-ratio) of 'train' sessions → test_normal.")
    eval_parser.add_argument("--model_dir", type=str, default=None,
                             help="Model directory to load best_bert.pth from.")

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')
    vocab_parser.add_argument("-s", "--vocab_size", type=int,  default=None)
    vocab_parser.add_argument("-e", "--encoding",   type=str,  default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq",   type=int,  default=1)

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == 'train':
        if hasattr(args, 'ratio') and args.ratio is not None:
            options["train_ratio"] = args.ratio
            print(f"[logbert_seq] train_ratio overridden → {args.ratio} (sequential)")
        if hasattr(args, 'model_dir') and args.model_dir is not None:
            options["model_dir"]  = args.model_dir.rstrip('/') + '/'
            options["model_path"] = options["model_dir"] + "best_bert.pth"
            options["scale_path"] = options["model_dir"] + "scale.pkl"
            print(f"[logbert_seq] model_dir overridden → {options['model_dir']}")
        Trainer(options).train()

    elif args.mode == 'predict':
        options["gaussian_mean"] = args.mean
        options["gaussian_std"]  = args.std
        Predictor(options).predict()

    elif args.mode == 'eval':
        import time as _time
        import torch
        from torch.utils.data import DataLoader
        from bert_pytorch.dataset import LogDataset
        from bert_pytorch.predict_log import compute_anomaly, find_best_threshold

        # Apply overrides
        if hasattr(args, 'ratio') and args.ratio is not None:
            options["train_ratio"] = args.ratio
        if hasattr(args, 'model_dir') and args.model_dir is not None:
            options["model_dir"]  = args.model_dir.rstrip('/') + '/'
            options["model_path"] = options["model_dir"] + "best_bert.pth"
            options["scale_path"] = options["model_dir"] + "scale.pkl"
        ratio = options["train_ratio"]
        print(f"[logbert_seq eval] ratio={ratio}  model_dir={options['model_dir']}")

        # Load model and vocab
        model = torch.load(options["model_path"])
        model.to(options["device"])
        model.eval()
        vocab = WordVocab.load_vocab(options["vocab_path"])

        center = radius = None
        if options["hypersphere_loss"]:
            center_dict = torch.load(options["model_dir"] + "best_center.pt")
            center = center_dict["center"]
            radius = center_dict["radius"]

        # test_normal  = last (1-ratio) sessions from the normal 'train' file
        # test_abnormal = the anomaly sessions file from data_process.py
        with open(options["train_vocab"], 'r') as f:
            all_normal = f.readlines()
        n_train_sessions  = int(len(all_normal) * ratio)
        test_normal_lines = all_normal[n_train_sessions:]

        with open(options["output_dir"] + "test_abnormal", 'r') as f:
            test_abnormal_lines = f.readlines()

        print(f"  test_normal   sessions : {len(test_normal_lines)}")
        print(f"  test_abnormal sessions : {len(test_abnormal_lines)}")

        def lines_to_windows(lines, desc):
            log_seqs, tim_seqs = [], []
            for line in tqdm(lines, desc=desc):
                ls, ts = fixed_window(line, options["window_size"],
                                      options["adaptive_window"],
                                      options["seq_len"], options["min_len"])
                if ls:
                    log_seqs += ls
                    tim_seqs += ts
            if not log_seqs:
                return np.array([], dtype=object), np.array([], dtype=object)
            log_seqs = np.array(log_seqs, dtype=object)
            tim_seqs = np.array(tim_seqs, dtype=object)
            idx = np.argsort(-np.array(list(map(len, log_seqs))))
            print(f"  {desc} windows : {len(log_seqs)}")
            return log_seqs[idx], tim_seqs[idx]

        def predict_windows(log_seqs, tim_seqs, desc):
            if len(log_seqs) == 0:
                return []
            seq_dataset = LogDataset(log_seqs, tim_seqs, vocab,
                                     seq_len=options["seq_len"],
                                     corpus_lines=options["corpus_lines"],
                                     on_memory=options["on_memory"],
                                     predict_mode=True,
                                     mask_ratio=options["mask_ratio"])
            loader = DataLoader(seq_dataset, batch_size=options["batch_size"],
                                num_workers=0, collate_fn=seq_dataset.collate_fn)
            results = []
            for data in tqdm(loader, desc=desc):
                data = {k: v.to(options["device"]) for k, v in data.items()}
                out  = model(data["bert_input"], data["time_input"])
                lm   = out["logkey_output"]   # (B, seq_len, vocab_size)

                for i in range(len(data["bert_label"])):
                    mask_idx  = data["bert_label"][i] > 0
                    n_masked  = mask_idx.sum().item()
                    n_undetected = 0
                    if options["is_logkey"] and n_masked > 0:
                        topk   = torch.argsort(-lm[i][mask_idx], dim=1)[:, :options["num_candidates"]]
                        labels = data["bert_label"][i][mask_idx]
                        n_undetected = int((labels.unsqueeze(1) != topk).all(dim=1).sum().item())

                    svdd_label = 0
                    if options["hypersphere_loss_test"] and center is not None:
                        dist = torch.sqrt(torch.sum((out["cls_output"][i] - center) ** 2))
                        svdd_label = int(dist.item() > radius)

                    results.append({
                        "num_error"        : 0,
                        "undetected_tokens": n_undetected,
                        "masked_tokens"    : n_masked,
                        "total_logkey"     : int((data["bert_input"][i] > 0).sum()),
                        "deepSVDD_label"   : svdd_label,
                    })
            return results

        t0 = _time.time()
        normal_wins,   normal_times   = lines_to_windows(test_normal_lines,   "windows normal")
        abnormal_wins, abnormal_times = lines_to_windows(test_abnormal_lines, "windows abnormal")

        normal_results   = predict_windows(normal_wins,   normal_times,   "predict normal")
        abnormal_results = predict_windows(abnormal_wins, abnormal_times, "predict abnormal")

        params = {"is_logkey"            : options["is_logkey"],
                  "is_time"              : options["is_time"],
                  "hypersphere_loss"     : options["hypersphere_loss"],
                  "hypersphere_loss_test": options["hypersphere_loss_test"]}

        _, best_seq_th, FP, TP, TN, FN, P, R, F1 = find_best_threshold(
            normal_results, abnormal_results, params=params,
            th_range=np.arange(10), seq_range=np.arange(0, 1, 0.1))

        elapsed = _time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"  Results  (ratio={ratio:.1f},  seq_th={best_seq_th:.1f})")
        print(f"{'=' * 60}")
        print(f"  TP={TP}   TN={TN}   FP={FP}   FN={FN}")
        print(f"  Precision : {P:.2f}%")
        print(f"  Recall    : {R:.2f}%")
        print(f"  F1 Score  : {F1:.2f}%")
        print(f"  Elapsed   : {elapsed:.1f}s")

    elif args.mode == 'vocab':
        with open(options["train_vocab"], 'r') as f:
            logs = f.readlines()
        vocab = WordVocab(logs)
        print("vocab_size", len(vocab))
        vocab.save_vocab(options["vocab_path"])
