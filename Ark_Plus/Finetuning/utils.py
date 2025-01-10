from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score, f1_score, matthews_corrcoef
import torch
import numpy as np

class MetricLogger(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressLogger(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def read_from_csv(csv_file):
    arr = []
    lines = open(csv_file).readlines()
    for line in lines[1:]:
        row = line.strip().split(",")
        row = [float(v) for v in row]
        arr.append(row)
    return np.array(arr)

def get_classwise_mean_std(data):
    data = np.array(data)
    class_wise_mean, class_wise_std = [],[]
    _, n_class = data.shape
    for ic in range(n_class):
        class_wise_mean.append(np.mean(data[:,ic]))
        class_wise_std.append(np.std(data[:,ic]))
    return [class_wise_mean, class_wise_std]

def meanMCC(ground_truth, predictions):
    thresholds_all = []
    ap_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            fpr, tpr, thresholds = roc_curve(ground_truth[:, i], predictions[:, i])
            youden_j = tpr - fpr
            optimal_threshold = thresholds[np.argmax(youden_j)]
            thresholds_all.append(optimal_threshold)
            binary_predictions = (predictions[:, i] > optimal_threshold).astype(int)
            ap = matthews_corrcoef(ground_truth[:, i], binary_predictions)
            ap_scores.append(ap)

    map_score = np.mean(ap_scores)
    print(thresholds_all)
    return map_score, ap_scores

def meanAP(ground_truth, predictions):
    # Compute mean Average Precision (mAP)
    ap_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            ap = average_precision_score(ground_truth[:, i], predictions[:, i])
            ap_scores.append(ap)

    map_score = np.mean(ap_scores)
    return map_score, ap_scores

def meanAUC(ground_truth, predictions):
    # Compute mean Area Under the ROC Curve (mAUC)
    auc_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            auc = roc_auc_score(ground_truth[:, i], predictions[:, i])
            auc_scores.append(auc)

    mauc_score = np.mean(auc_scores)
    return mauc_score, auc_scores
def meanF1(ground_truth, predictions):
    # Compute mean F1 score (mF1)
    f1_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            fpr, tpr, thresholds = roc_curve(ground_truth[:, i], predictions[:, i])
            youden_j = tpr - fpr
            optimal_threshold = thresholds[np.argmax(youden_j)]
            binary_predictions = (predictions[:, i] > optimal_threshold).astype(int)
            f1 = f1_score(ground_truth[:, i], binary_predictions)
            f1_scores.append(f1)

    mf1_score = np.mean(f1_scores)
    return mf1_score, f1_scores

def metric_AUROC(target, output, nb_classes=14):
    outAUROC = []

    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for i in range(nb_classes):
        outAUROC.append(roc_auc_score(target[:, i], output[:, i]))

    return outAUROC


def vararg_callback_bool(option, opt_str, value, parser):
    assert value is None

    arg = parser.rargs[0]
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        value = True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        value = False

    del parser.rargs[:1]
    setattr(parser.values, option.dest, value)


def vararg_callback_int(option, opt_str, value, parser):
    assert value is None
    value = []

    def intable(str):
        try:
            int(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not intable(arg):
            break
        value.append(int(arg))

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

def step_decay(step, lr, epochs):

    progress = (step - 20) / float(epochs - 20)
    progress = np.clip(progress, 0.0, 1.0)
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))

    lr = lr * np.minimum(1., step / 20)

    return lr

def cosine_anneal_schedule(t,epochs,learning_rate):
    T=epochs
    M=1
    alpha_zero = learning_rate

    cos_inner = np.pi * (t % (T // M))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    return float(alpha_zero / 2 * cos_out)

def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def mean_dice_coef(y_true,y_pred):
    sum=0
    for i in range (y_true.shape[0]):
        sum += dice(y_true[i,:,:,:],y_pred[i,:,:,:])
    return sum/y_true.shape[0]

def load_swin_pretrained(ckpt, model):
    state_dict = ckpt
    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......", file=writter)
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero

    # if 'head.bias' in state_dict:
    #     head_bias_pretrained = state_dict['head.bias']
    #     Nc1 = head_bias_pretrained.shape[0]
    # else:
    #     Nc1 = -1
    # Nc2 = model.head.bias.shape[0]
    #
    # if (Nc1 != Nc2):
    #     if Nc1 == 21841 and Nc2 == 1000:
    #         print("loading ImageNet-22K weight to ImageNet-1K ......", file=writter)
    #         map22kto1k_path = f'data/map22kto1k.txt'
    #         with open(map22kto1k_path) as f:
    #             map22kto1k = f.readlines()
    #         map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
    #         state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
    #         state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
    #     else:
    #         torch.nn.init.constant_(model.head.bias, 0.)
    #         torch.nn.init.constant_(model.head.weight, 0.)
    #         if Nc1 != -1:
    #             del state_dict['head.weight']
    #             del state_dict['head.bias']
    #         print(f"Error in loading classifier head, re-init classifier head to 0", file=writter)



    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    del ckpt
    torch.cuda.empty_cache()
