import numpy as np
import torch


def sweep_thresholds_curves(pred, gt, thres_range):
    pr, roc, f1 = [], [], []
    for thres in thres_range:
        true_positives, false_positives, true_negatives, false_negatives = get_stats(pred, gt, thres)

        tpr = np.nan_to_num(true_positives / (true_positives + false_negatives), nan=0)
        fpr = np.nan_to_num(false_positives / (true_negatives + false_positives), nan=0)

        precision = np.nan_to_num(true_positives / (false_positives + true_positives), nan=1)
        recall = tpr
        f1_score = np.nan_to_num(2 * (precision * recall) / (precision + recall), 0)

        pr.append(np.stack([recall, precision]))
        roc.append(np.stack([fpr, tpr]))
        f1.append(f1_score)

    optimal_thresholds = thres_range[np.argmax(f1, axis=0)]

    return np.stack(pr), np.stack(roc), np.stack(f1), optimal_thresholds


def get_performance_mesaures(pred, gt, thres):
    true_positives, false_positives, true_negatives, false_negatives = get_stats(pred, gt, thres)

    accuracy = (true_positives + true_negatives) / (false_positives + false_negatives + true_positives + true_negatives)
    precision = true_positives / (false_positives + true_positives)
    recall = true_positives / (false_negatives + true_positives)

    f1 = 2 * (precision * recall) / (precision + recall)
    macro_f1 = 2 * (precision.nanmean() * recall.nanmean()) / (precision.nanmean() + recall.nanmean())

    return accuracy, precision, recall, f1, macro_f1


def get_stats(pred, gt, thres):
    binary_pred = get_binary_prediction(pred, thres)

    # TODO: REALLY DELETE
    normal_pred = (binary_pred.sum(dim=1) == 0).float().unsqueeze(dim=1)
    normal_gt = (gt.sum(dim=1) == 0).float().unsqueeze(dim=1)

    with_normal_pred = torch.concat([binary_pred, normal_pred], dim=1)
    with_normal_gt = torch.concat([gt, normal_gt], dim=1)
    confusion = with_normal_pred / with_normal_gt
    # TODO: END REALLY DELETE

    # confusion = binary_pred / gt  # TODO: REALLY UNCOMMENT
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion == 1, dim=0)
    false_positives = torch.sum(confusion == float('inf'), dim=0)
    true_negatives = torch.sum(torch.isnan(confusion), dim=0)
    false_negatives = torch.sum(confusion == 0, dim=0)

    return true_positives, false_positives, true_negatives, false_negatives


def get_binary_prediction(pred, thres):
    return (torch.sigmoid(pred) > thres).float()
