import os
import random
import time
from datetime import datetime

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torchnet import meter
from tqdm import tqdm

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from model.equisite_t3_pro import EquiSite
from utils.loss import CB_loss, TripletCenterLoss
from utils.valid_metrices import CFM_eval_metrics

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed, deterministic=True):
    """Seed Python, NumPy, PyTorch and CUDA random number generators.

    Args:
        seed (int): Global random seed.
        deterministic (bool): Use deterministic cuDNN settings and request
            deterministic PyTorch algorithms where supported. Unsupported
            deterministic operations emit warnings instead of terminating the
            run, which keeps compatibility with PyG/equivariant CUDA kernels.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Set this before any CUDA operation. It is required by deterministic
        # CUDA matrix multiplication on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # warn_only=True avoids hard failures when a PyG/equivariant operation
        # has no deterministic CUDA implementation in the installed version.
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch versions may not support the warn_only argument.
                pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(False)
            except TypeError:
                pass


def seed_worker(worker_id):
    """Seed NumPy and Python RNGs inside each DataLoader worker."""
    del worker_id  # The worker seed is already encoded in torch.initial_seed().
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def find_best_mcc_threshold(labels, preds):
    """Select a probability threshold that maximizes MCC in O(n log n)."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    preds = np.asarray(preds).reshape(-1)

    order = np.argsort(preds)[::-1]
    sorted_labels = labels[order]
    sorted_preds = preds[order]
    distinct = np.r_[np.where(np.diff(sorted_preds))[0], len(sorted_preds) - 1]

    tp = np.cumsum(sorted_labels)[distinct].astype(np.float64)
    fp = distinct.astype(np.float64) + 1 - tp
    positives = float(sorted_labels.sum())
    negatives = float(len(sorted_labels) - sorted_labels.sum())
    fn = positives - tp
    tn = negatives - fp

    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(tp),
        where=denominator != 0,
    )
    return float(sorted_preds[distinct[int(np.argmax(mcc))]])


def train(args, model, loader, optimizer, tcl, device):
    model.train()
    tcl.train()
    loss_accum = 0
    preds_bi, preds = [], []
    labels = []

    for step, batch in enumerate(tqdm(loader, disable=args.disable_tqdm)):
        batch = batch.to(device)
        try:
            if args.equiformer:
                pred, logit, emb = model(batch)
            else:
                pred, logit, emb = model(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                print('\n forward error \n')
                raise e
            else:
                print('OOM')
                torch.cuda.empty_cache()
                continue

        label = batch.y
        optimizer.zero_grad()

        if args.loss_type == "MSE":
            label_onehot = F.one_hot(
                label.squeeze().to(torch.int64), num_classes=2
            ).float()
            criterion = nn.MSELoss()
            loss = criterion(pred, label_onehot)
        elif args.loss_type == "BCE":
            criterion = nn.BCELoss()
            loss = criterion(pred, label.squeeze())
        elif args.loss_type == "FocalLoss":
            beta, gamma = 0.999, 2.0
            num_ones = torch.count_nonzero(label).item()
            num_zeros = label.numel() - num_ones
            samples_per_cls = [num_zeros, num_ones]
            loss_fl = CB_loss(
                label, logit, samples_per_cls, 2, "focal", beta, gamma
            )
            loss_tcl = tcl(emb, label)
            loss = loss_fl + 0.1 * loss_tcl

        loss.backward()
        optimizer.step()

        labels.append(label)
        preds.append(pred)
        loss_accum += loss.item()

    labels = torch.cat(labels, dim=0).squeeze().cpu()
    preds = torch.cat(preds, dim=0).squeeze().cpu().detach().numpy()

    # ROC_AUC
    roc_auc = roc_auc_score(labels, preds)

    # PR_AUC
    precision, recall, thresholds_pr = precision_recall_curve(labels, preds)
    pr_auc_1 = auc(recall, precision)
    pr_auc_2 = average_precision_score(labels, preds)

    # Select the MCC threshold only from training predictions.
    best_threshold_pr = find_best_mcc_threshold(labels, preds)
    preds_bi = (preds >= best_threshold_pr).astype(np.int64)
    pre = precision_score(labels, preds_bi, zero_division=0)
    rec = recall_score(labels, preds_bi, zero_division=0)
    f1 = f1_score(labels, preds_bi, zero_division=0)
    mcc = matthews_corrcoef(labels, preds_bi)

    return (
        loss_accum / (step + 1),
        pr_auc_1,
        pr_auc_2,
        roc_auc,
        pre,
        rec,
        f1,
        mcc,
        best_threshold_pr,
    )


def evaluation(args, model, loader, val_th, device):
    model.eval()
    loss_accum = 0

    if val_th is not None:
        preds, labels = [], []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                batch = batch.to(device)
                try:
                    pred, logit, emb = model(batch)
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        print('\n forward error \n')
                        raise e
                    else:
                        print('evaluation OOM')
                        torch.cuda.empty_cache()
                        continue

                label = batch.y
                criterion = nn.BCELoss()
                loss = criterion(pred, label.squeeze())
                loss_accum += loss.item()
                labels.append(label)
                preds.append(pred)
                torch.cuda.empty_cache()

        labels = torch.cat(labels, dim=0).squeeze().cpu()
        preds = torch.cat(preds, dim=0).squeeze().cpu().detach().numpy()

        # ROC_AUC
        roc_auc = roc_auc_score(labels, preds)

        # PR_AUC
        precision, recall, thresholds_pr = precision_recall_curve(labels, preds)
        pr_auc_1 = auc(recall, precision)
        pr_auc_2 = average_precision_score(labels, preds)

        # Precision, recall, F1 and MCC all use the supplied training threshold.
        best_threshold_pr = val_th
        preds_bi = (preds >= best_threshold_pr).astype(np.int64)
        pre = precision_score(labels, preds_bi, zero_division=0)
        rec = recall_score(labels, preds_bi, zero_division=0)
        f1 = f1_score(labels, preds_bi, zero_division=0)
        mcc = matthews_corrcoef(labels, preds_bi)
    else:
        # Kept for compatibility with callers that intentionally want to select
        # a threshold on the current loader. main() does not use this branch for
        # the test set.
        preds, labels = [], []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                batch = batch.to(device)
                try:
                    pred, logit, emb = model(batch)
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        print('\n forward error \n')
                        raise e
                    else:
                        print('evaluation OOM')
                        torch.cuda.empty_cache()
                        continue

                label = batch.y
                criterion = nn.BCELoss()
                loss = criterion(pred, label.squeeze())
                loss_accum += loss.item()
                labels.append(label)
                preds.append(pred)

        labels = torch.cat(labels, dim=0).squeeze().cpu()
        preds = torch.cat(preds, dim=0).squeeze().cpu().detach().numpy()

        # ROC_AUC
        roc_auc = roc_auc_score(labels, preds)

        # PR_AUC
        precision, recall, thresholds_pr = precision_recall_curve(labels, preds)
        pr_auc_1 = auc(recall, precision)
        pr_auc_2 = average_precision_score(labels, preds)

        best_threshold_pr = find_best_mcc_threshold(labels, preds)
        preds_bi = (preds >= best_threshold_pr).astype(np.int64)
        pre = precision_score(labels, preds_bi, zero_division=0)
        rec = recall_score(labels, preds_bi, zero_division=0)
        f1 = f1_score(labels, preds_bi, zero_division=0)
        mcc = matthews_corrcoef(labels, preds_bi)

    return (
        loss_accum / (step + 1),
        pr_auc_1,
        pr_auc_2,
        roc_auc,
        pre,
        rec,
        f1,
        mcc,
        best_threshold_pr,
    )


def main():
    ### Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='Device to use')
    parser.add_argument(
        '--num_workers', type=int, default=6,
        help='Number of workers in Dataloader'
    )

    ### Data
    parser.add_argument(
        '--dataset', type=str, default='DNA_Check',
        help='DNA, RNA, ATP, HEME'
    )
    parser.add_argument(
        '--dataset_path', type=str, default='dataset/',
        help='path to load and process the data'
    )

    # data augmentation tricks, see appendix E in the paper
    # (https://openreview.net/pdf?id=9X-hgLDLYkQ)
    parser.add_argument(
        '--mask', action='store_true', help='Random mask some node type'
    )
    parser.add_argument(
        '--noise', action='store_true',
        help='Add Gaussian noise to node coords'
    )
    parser.add_argument(
        '--deform', action='store_true', help='Deform node coords'
    )
    parser.add_argument(
        '--data_augment_eachlayer', action='store_true',
        help='Add Gaussian noise to features'
    )
    parser.add_argument(
        '--euler_noise', action='store_true',
        help='Add Gaussian noise Euler angles'
    )
    parser.add_argument(
        '--mask_aatype', type=float, default=0.1,
        help='Random mask aatype to 25(unknown:X) ratio'
    )

    ### Model
    parser.add_argument(
        '--level', type=str, default='allatom+esm',
        help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels'
    )
    parser.add_argument(
        '--num_blocks', type=int, default=4, help='Model layers, 4'
    )
    parser.add_argument(
        '--hidden_channels', type=int, default=128,
        help='Hidden dimension'
    )
    parser.add_argument(
        '--out_channels', type=int, default=1,
        help=''
    )
    parser.add_argument('--fix_dist', action='store_true')
    parser.add_argument(
        '--cutoff', type=float, default=11.5,
        help='Distance constraint for building the protein graph'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.25, help='Dropout'
    )

    ### Training hyperparameter
    parser.add_argument(
        '--epochs', type=int, default=200,
        help='Number of epochs to train'
    )
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument(
        '--lr_decay_step_size', type=int, default=60,
        help='Learning rate step size'
    )
    parser.add_argument(
        '--lr_decay_factor', type=float, default=0.5,
        help='Learning rate factor'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0, help='Weight Decay'
    )
    parser.add_argument(
        '--batch_size', type=int, default=2,
        help='Batch size during training'
    )
    parser.add_argument(
        '--eval_batch_size', type=int, default=2, help='Batch size'
    )
    parser.add_argument(
        '--loss_type', type=str, default="FocalLoss",
        choices=["BCE", "MSE", "FocalLoss"]
    )
    parser.add_argument('--equiformer', type=bool, default=True)
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument(
        '--save_dir', type=str, default=None, help='Trained model path'
    )
    parser.add_argument(
        '--disable_tqdm', default=False, action='store_true'
    )

    # Reproducibility. The seed is active by default; use --non_deterministic
    # only when maximum speed is preferred over deterministic execution.
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Global random seed (default: 42)'
    )
    parser.add_argument(
        '--deterministic', dest='deterministic', action='store_true',
        help='Enable deterministic training behavior'
    )
    parser.add_argument(
        '--non_deterministic', dest='deterministic', action='store_false',
        help='Disable deterministic backend settings'
    )
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    # Must be called before dataset/DataLoader/model creation.
    set_seed(args.seed, deterministic=args.deterministic)
    print(args)
    print(
        'Random seed: {} | deterministic: {}'.format(
            args.seed, args.deterministic
        )
    )

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if args.dataset == 'DNA_Check':
        from dataset.DNA_Check.PBdataset import DBdataset
    elif args.dataset == 'RNA_Check':
        from dataset.RNA_Check.PBdataset import DBdataset
    elif args.dataset == 'PATP':
        from dataset.PATP.PBdataset import DBdataset
    elif args.dataset == 'PCA':
        from dataset.PCA.PBdataset import DBdataset
    elif args.dataset == 'PHEM':
        from dataset.PHEM.PBdataset import DBdataset
    elif args.dataset == 'PMG':
        from dataset.PMG.PBdataset import DBdataset
    elif args.dataset == 'PMN':
        from dataset.PMN.PBdataset import DBdataset
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset}")

    try:
        train_set = DBdataset(
            root=args.dataset_path + args.dataset, split='Train'
        )
        test_set = DBdataset(
            root=args.dataset_path + args.dataset, split='Test'
        )
    except FileNotFoundError:
        print('\n Please download data firstly')
        raise FileNotFoundError

    # Separate generators prevent iteration of one loader from changing the
    # random sequence used by the other loader.
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    test_generator = torch.Generator()
    test_generator.manual_seed(args.seed + 1)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=test_generator,
    )

    print('Done!')
    print('Train, val, test:', train_set, test_set)

    model = EquiSite(
        num_blocks=args.num_blocks,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        cutoff=args.cutoff,
        dropout=args.dropout,
        data_augment_eachlayer=args.data_augment_eachlayer,
        euler_noise=args.euler_noise,
        level=args.level,
        args=args,
    )
    model.to(device)

    # The model returns a 32-dimensional embedding for TCL.
    tcl = TripletCenterLoss(
        margin=5, num_classes=2, center_embed=32
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(tcl.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step_size,
        gamma=args.lr_decay_factor,
    )

    if args.continue_training:
        save_dir = args.save_dir
        checkpoint = torch.load(save_dir + '/best_val.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'tcl_state_dict' in checkpoint:
            tcl.load_state_dict(checkpoint['tcl_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print('Legacy checkpoint: TCL and optimizer states are reinitialized.')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        save_dir = (
            './saves/trained_models_{dataset}/{level}/'
            'layer{num_blocks}_cutoff{cutoff}_hidden{hidden_channels}_'
            'batch{batch_size}_lr{lr}_{lr_decay_factor}_'
            '{lr_decay_step_size}_dropout{dropout}__seed{seed}_{time}'
        ).format(
            dataset=args.dataset,
            level=args.level,
            num_blocks=args.num_blocks,
            cutoff=args.cutoff,
            hidden_channels=args.hidden_channels,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            dropout=args.dropout,
            seed=args.seed,
            time=datetime.now(),
        )
        print('saving to...', save_dir)
        start_epoch = 1

    num_params = sum(p.numel() for p in model.parameters())
    print('num_parameters:', num_params)
    writer = SummaryWriter(log_dir=save_dir)
    best_val_auc = 0

    for epoch in range(start_epoch, args.epochs + 1):
        print('==== Epoch {} ===='.format(epoch))
        (
            train_loss,
            train_pr_auc_1,
            train_pr_auc_2,
            train_roc_auc,
            train_precision,
            train_recall,
            train_f1,
            train_mcc,
            train_th,
        ) = train(args, model, train_loader, optimizer, tcl, device)

        (
            val_loss,
            val_pr_auc_1,
            val_pr_auc_2,
            val_roc_auc,
            val_precision,
            val_recall,
            val_f1,
            val_mcc,
            val_th,
        ) = evaluation(
            args, model, test_loader, val_th=train_th, device=device
        )

        if not save_dir == "" and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not save_dir == "" and val_roc_auc > best_val_auc:
            print('Saving best val checkpoint ...')
            checkpoint = {
                'epoch': epoch,
                'seed': args.seed,
                'deterministic': args.deterministic,
                'model_state_dict': model.state_dict(),
                'tcl_state_dict': tcl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, save_dir + '/best_val.pt')
            best_val_auc = val_roc_auc

        print(
            "Train:",
            "\n",
            "ROC_AUC: {:.4f}, PR_AUC:{:.4f}, PR_AUC:{:.4f}, "
            "Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, "
            "MCC: {:.4f}, Train_th: {}".format(
                train_roc_auc,
                train_pr_auc_1,
                train_pr_auc_2,
                train_precision,
                train_recall,
                train_f1,
                train_mcc,
                train_th,
            ),
            sep="",
        )
        print(
            "Test:",
            "\n",
            "ROC_AUC: {:.4f}, PR_AUC:{:.4f}, PR_AUC:{:.4f}, "
            "Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, "
            "MCC: {:.4f}, Train_th: {}".format(
                val_roc_auc,
                val_pr_auc_1,
                val_pr_auc_2,
                val_precision,
                val_recall,
                val_f1,
                val_mcc,
                train_th,
            ),
            sep="",
        )
        print("test_auc@best_val:{:.4f}".format(best_val_auc))

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_roc_auc', train_roc_auc, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_roc_auc', val_roc_auc, epoch)
        scheduler.step()

    writer.close()

    # Save last model
    checkpoint = {
        'epoch': epoch,
        'seed': args.seed,
        'deterministic': args.deterministic,
        'model_state_dict': model.state_dict(),
        'tcl_state_dict': tcl.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, save_dir + "/epoch{}.pt".format(epoch))


if __name__ == "__main__":
    main()
