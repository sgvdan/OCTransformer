import torch


def get_stats(pred, gt, thres):
    binary_pred = (pred > thres).float()
    confusion = binary_pred / gt
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

    accuracy = (true_positives + true_negatives) / (false_positives + false_negatives + true_positives + true_negatives)
    precision = true_positives / (false_positives + true_positives)
    recall = true_positives / (false_negatives + true_positives)

    f1 = 2 * (precision * recall) / (precision + recall)
    macro_f1 = 2 * (precision.nanmean() * recall.nanmean()) / (precision.nanmean() + recall.nanmean())

    return accuracy, precision, recall, f1, macro_f1
