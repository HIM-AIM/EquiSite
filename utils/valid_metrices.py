import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, matthews_corrcoef
def CFM_eval_metrics(CFM):
    CFM = CFM.astype(float)
    tn = CFM[0, 0]
    fp = CFM[0, 1]
    fn = CFM[1, 0]
    tp = CFM[1, 1]
    if tp > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0
    if tp > 0:
        pre = tp / (tp + fp)
    else:
        pre = 0
    if tn > 0:
        spe = tn / (tn + fp)
    else:
        spe = 0
    if rec + pre > 0:
        F1 = 2 * rec * pre / (rec + pre)
    else:
        F1 = 0
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    else:
        mcc = -1
    return rec, pre,F1, spe, mcc


def best_threshold_by_mcc(labels, preds):
    # 计算 precision, recall 和 阈值
    precision, recall, thresholds = precision_recall_curve(labels, preds)

    # 初始化最大MCC和最佳阈值
    best_mcc = -1
    best_threshold = 0

    # 为了包含所有阈值，我们在thresholds的头部添加一个0
    thresholds = np.insert(thresholds, 0, 0)

    # 遍历每个阈值，计算对应的MCC
    for threshold in thresholds:
        # 根据阈值将预测概率转换为二进制预测
        binary_preds = (preds >= threshold).astype(int)

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()

        # 计算MCC
        mcc = matthews_corrcoef(labels, binary_preds)

        # 更新最大MCC和最佳阈值
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    return best_threshold, best_mcc

