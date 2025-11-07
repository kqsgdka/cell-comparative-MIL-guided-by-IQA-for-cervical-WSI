import numpy as np

def _prf_divide(numerator, denominator, metric,
                modifier, average, warn_for, zero_division="warn"):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.

    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ["warn", 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != "warn" or metric not in warn_for:
        return result

    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."

    if metric in warn_for and 'f-score' in warn_for:
        msg_start = '{0} and F-score are'.format(metric.title())
    elif metric in warn_for:
        msg_start = '{0} is'.format(metric.title())
    elif 'f-score' in warn_for:
        msg_start = 'F-score is'
    else:
        return result

    print(average, modifier, msg_start, len(result))

    return result

def precision_recall_fscore_support(MCM, beta=1.0, labels=None,
                                    pos_label=1, average=None,
                                    warn_for=('precision', 'recall',
                                              'f-score'),
                                    sample_weight=None,
                                    zero_division="warn"):
    """Compute precision, recall, F-measure and support for each class.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_fscore_support
    >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    (0.22..., 0.33..., 0.26..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    (0.33..., 0.33..., 0.33..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    (0.22..., 0.33..., 0.26..., None)

    It is possible to compute per-label precisions, recalls, F1-scores and
    supports instead of averaging:

    >>> precision_recall_fscore_support(y_true, y_pred, average=None,
    ... labels=['pig', 'dog', 'cat'])
    (array([0.        , 0.        , 0.66...]),
     array([0., 0., 1.]), array([0. , 0. , 0.8]),
     array([2, 2, 2]))
    """
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")

    samplewise = average == 'samples'

    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(tp_sum, pred_sum, 'precision',
                            'predicted', average, warn_for, zero_division)
    recall = _prf_divide(tp_sum, true_sum, 'recall',
                         'true', average, warn_for, zero_division)

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined

                  
    if np.isposinf(beta):
        f_score=recall
    else:
        denom=beta2 * precision + recall

        denom[denom == 0.]=1  # avoid division by 0
        f_score=(1 + beta2) * precision * recall / denom

    # Average the results
    if average == 'weighted':
        weights=true_sum
        if weights.sum() == 0:
            zero_division_value=np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value=np.float64(0.0)
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            if pred_sum.sum() == 0:
                return (zero_division_value,
                        zero_division_value,
                        zero_division_value,
                        None)
            else:
                return (np.float64(0.0),
                        zero_division_value,
                        np.float64(0.0),
                        None)

    elif average == 'samples':
        weights=sample_weight
    else:
        weights=None

    if average is not None:
        assert average != 'binary' or len(precision) == 1
        precision=np.average(precision, weights=weights)
        recall=np.average(recall, weights=weights)
        f_score=np.average(f_score, weights=weights)
        true_sum=None  # return no support

    return precision, recall, f_score, true_sum
    




if __name__ == "__main__":
    import torch
    # x=torch.tensor([1,2,3,1,1,1,2,3,3,3,3,3])
    # y=torch.tensor([1,2,3,1,1,1,1,1,1,1,2,3])
    # cm=confusion_matrix(x,y)
    # label=["cat","dog","bird"]
    # plot_confusion_matrix(cm,"/home/qinjian/分类网络/Abnormal_cell_classification/1.png",label)
    # from sklearn.metrics import classification_report
    # report=classification_report(x,y,target_names=label,digits=4)
    # print(report)

    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    cm=np.array([[[158,3,0,0,1], [2,159,2,0,2], [0,4,154,0,0],[0,0,0,157,0],[0,0,0,0,166]]])
    print(precision_recall_fscore_support(cm,average='micro'))
    label=["cat", "dog", "bird"]

