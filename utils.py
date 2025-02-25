import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

def accuracy(output, target, topk=(1,)):
    """Calculate accuracy for the specified top-k predictions"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def AA_andEachClassAccuracy(confusion_matrix):
    """Calculate average accuracy and per-class accuracy"""
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(list_diag / list_raw_sum)
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports(y_pred, y_test, dataset_name):
    """Generate comprehensive classification reports"""
    classification = classification_report(y_test, y_pred, digits=6)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    results = list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))
    return classification, confusion, results

def adjust_learning_rate(optimizer, epoch, config):
    """Adjust learning rate based on epoch"""
    lr = config.LEARNING_RATE * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr