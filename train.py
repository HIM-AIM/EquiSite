import os
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from model.equisite_t3_pro import EquiSite


from torch_geometric.data import DataLoader

from torchnet import meter
from sklearn.metrics import *
from utils.loss import CB_loss, TripletCenterLoss
from utils.valid_metrices import CFM_eval_metrics
import warnings

warnings.filterwarnings("ignore")


# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.random.manual_seed(seed)
#     # dgl.random.seed(seed)
#     if torch.cuda.is_available():
#        torch.cuda.manual_seed(seed)
#        torch.cuda.manual_seed_all(seed)
def train(args, model, loader, optimizer, device):
    model.train()
    tcl = TripletCenterLoss(margin=5, num_classes=2)
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
                raise (e)
            else:
                print('OOM')
            torch.cuda.empty_cache()
            continue
        label = batch.y
        optimizer.zero_grad()
        if args.loss_type == "MSE":
            label_onehot = F.one_hot(label.squeeze().to(torch.int64), num_classes=2).float()
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
            loss_fl = CB_loss(label, logit, samples_per_cls, 2, "focal", beta, gamma)
            loss_tcl = tcl(logit, label)
            loss = loss_fl + 0.1*loss_tcl
        loss.backward()
        optimizer.step()
        labels.append(label)
        preds.append(pred)
        loss_accum += loss.item()
    labels, preds = torch.cat(labels, dim=0).squeeze().cpu(), torch.cat(preds, dim=0).squeeze().cpu().detach().numpy()
    # ROC_AUC
    roc_auc = roc_auc_score(labels, preds)
    # PR_AUC
    precision, recall, thresholds_pr = precision_recall_curve(labels, preds)
    pr_auc_1 = auc(recall, precision)
    pr_auc_2 = average_precision_score(labels, preds)
    # MCC find thereshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1 = max(f1_scores)
    best_threshold_pr = thresholds_pr[np.argmax(f1_scores)]
    # pre, rec, mcc
    preds_bi = [1 if p >= best_threshold_pr else 0 for p in preds]
    pre = precision_score(labels, preds_bi)
    rec = recall_score(labels, preds_bi)
    mcc = matthews_corrcoef(labels, preds_bi)

    return loss_accum / (step + 1), pr_auc_1, pr_auc_2, roc_auc, pre, rec, f1, mcc, best_threshold_pr


def evaluation(args, model, loader, val_th, device):
    model.eval()
    loss_accum = 0
    if val_th is not None:
        preds, labels = [], []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                batch = batch.to(device)
                # pred = model(batch)
                try:
                    pred, logit, emb = model(batch)
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        print('\n forward error \n')
                        raise (e)
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
            labels, preds = torch.cat(labels, dim=0).squeeze().cpu(), \
                            torch.cat(preds,dim=0).squeeze().cpu().detach().numpy()
            # ROC_AUC
            roc_auc = roc_auc_score(labels, preds)
            # PR_AUC
            precision, recall, thresholds_pr = precision_recall_curve(labels, preds)
            pr_auc_1 = auc(recall, precision)
            pr_auc_2 = average_precision_score(labels, preds)
            # pre, rec, f1, mcc
            preds_bi = [1 if p >= val_th else 0 for p in preds]
            pre = precision_score(labels, preds_bi)
            rec = recall_score(labels, preds_bi)
            f1 = 2 * (pre * rec) / (pre + rec)
            mcc = matthews_corrcoef(labels, preds_bi)
    else:

        preds, labels = [], []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                batch = batch.to(device)
                # pred = model(batch)
                try:
                    pred, logit, emb = model(batch)
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        print('\n forward error \n')
                        raise (e)
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

            labels, preds = torch.cat(labels, dim=0).squeeze().cpu(), torch.cat(preds,dim=0).squeeze().cpu().detach().numpy()
            # ROC_AUC
            roc_auc = roc_auc_score(labels, preds)
            # PR_AUC
            precision, recall, thresholds_pr = precision_recall_curve(labels, preds)
            pr_auc_1 = auc(recall, precision)
            pr_auc_2 = average_precision_score(labels, preds)
            # MCC find thereshold
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1 = max(f1_scores)
            best_threshold_pr = thresholds_pr[np.argmax(f1_scores)]
            #pre, rec, mcc
            preds_bi = [1 if p >= best_threshold_pr else 0 for p in preds]
            pre = precision_score(labels, preds_bi)
            rec = recall_score(labels, preds_bi)
            mcc = matthews_corrcoef(labels, preds_bi)
    return loss_accum / (step + 1), pr_auc_1, pr_auc_2, roc_auc, pre, rec, f1, mcc, best_threshold_pr


def main():
    ### Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0, help='Device to use')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers in Dataloader')

    ### Data
    parser.add_argument('--dataset', type=str, default='RNA_Fix', help='DNA, RNA, ATP, HEME')
    parser.add_argument('--dataset_path', type=str, default='dataset/', help='path to load and process the data')

    # data augmentation tricks, see appendix E in the paper (https://openreview.net/pdf?id=9X-hgLDLYkQ)
    parser.add_argument('--mask', action='store_true', help='Random mask some node type')
    parser.add_argument('--noise', action='store_true', help='Add Gaussian noise to node coords')
    parser.add_argument('--deform', action='store_true', help='Deform node coords')
    parser.add_argument('--data_augment_eachlayer', action='store_true', help='Add Gaussian noise to features')
    parser.add_argument('--euler_noise', action='store_true', help='Add Gaussian noise Euler angles')
    parser.add_argument('--mask_aatype', type=float, default=0.1, help='Random mask aatype to 25(unknown:X) ratio')

    ### Model
    parser.add_argument('--level', type=str, default='allatom+esm',
                        help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels')
    parser.add_argument('--num_blocks', type=int, default=4, help='Model layers, 4')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--out_channels', type=int, default=1,
                        help='Number of classes, 1195 for the fold data, 384 for the ECdata')
    parser.add_argument('--fix_dist', action='store_true')
    parser.add_argument('--cutoff', type=float, default=11.5, help='Distance constraint for building the protein graph')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout')

    ### Training hyperparameter
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lr_decay_step_size', type=int, default=60, help='Learning rate step size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--loss_type', type=str, default="FocalLoss", choices=["BCE", "MSE", "FocalLoss"])

    parser.add_argument('--equiformer', type=bool, default=True)

    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None, help='Trained model path')

    parser.add_argument('--disable_tqdm', default=False, action='store_true')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


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
        train_set = DBdataset(root=args.dataset_path + args.dataset, split='Train')
        test_set = DBdataset(root=args.dataset_path + args.dataset, split='Test')
    except FileNotFoundError:
        print(
            '\n Please download data firstly, following https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph/dataset#ecdataset-and-folddataset and https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets \n')
        raise (FileNotFoundError)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    print('Done!')
    print('Train, val, test:', train_set, test_set)

    model = EquiSite(num_blocks=args.num_blocks, hidden_channels=args.hidden_channels, out_channels=args.out_channels,
                   cutoff=args.cutoff, dropout=args.dropout,
                   data_augment_eachlayer=args.data_augment_eachlayer,
                   euler_noise=args.euler_noise, level=args.level, args=args)
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)

    if args.continue_training:
        save_dir = args.save_dir
        checkpoint = torch.load(save_dir + '/best_val.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        save_dir = './saves/trained_models_{dataset}/{level}/layer{num_blocks}_cutoff{cutoff}_hidden{hidden_channels}_batch{batch_size}_lr{lr}_{lr_decay_factor}_{lr_decay_step_size}_dropout{dropout}__{time}'.format(
            dataset=args.dataset, level=args.level,
            num_blocks=args.num_blocks, cutoff=args.cutoff, hidden_channels=args.hidden_channels,
            batch_size=args.batch_size,
            lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size,
            dropout=args.dropout, time=datetime.now())
        print('saving to...', save_dir)
        start_epoch = 1

    num_params = sum(p.numel() for p in model.parameters())
    print('num_parameters:', num_params)


    writer = SummaryWriter(log_dir=save_dir)
    best_val_auc = 0


    for epoch in range(start_epoch, args.epochs + 1):
        print('==== Epoch {} ===='.format(epoch))


        train_loss, train_pr_auc_1, train_pr_auc_2, train_roc_auc, \
        train_precision, train_recall, train_f1, train_mcc, train_th = \
            train(args, model, train_loader, optimizer, device)

        val_loss, val_pr_auc_1, val_pr_auc_2, val_roc_auc, \
        val_precision, val_recall, val_f1, val_mcc, val_th = \
            evaluation(args, model, test_loader, val_th=None, device=device)


        if not save_dir == "" and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not save_dir == "" and val_roc_auc > best_val_auc:
            print('Saving best val checkpoint ...')
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict()}
            torch.save(checkpoint, save_dir + '/best_val.pt')
            best_val_auc = val_roc_auc


        print("Train:", "\n",
              "ROC_AUC: {:.4f}, PR_AUC:{:.4f}, PR_AUC:{:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, MCC: {:.4f}, Train_th: {}"
              .format(train_roc_auc, train_pr_auc_1, train_pr_auc_2,
                      train_precision, train_recall, train_f1, train_mcc, train_th,), sep="")
        print("Test:", "\n",
              "ROC_AUC: {:.4f}, PR_AUC:{:.4f}, PR_AUC:{:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, MCC: {:.4f}, Val_th: {}"
              .format(val_roc_auc, val_pr_auc_1, val_pr_auc_2,
                      val_precision, val_recall, val_f1, val_mcc, val_th,), sep="")
        print("test_auc@best_val:{:.4f}".format(best_val_auc))

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_roc_auc', train_roc_auc, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_roc_auc', val_roc_auc, epoch)


        scheduler.step()


    writer.close()
    # Save last model
    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
    torch.save(checkpoint, save_dir + "/epoch{}.pt".format(epoch))


if __name__ == "__main__":
    main()