import torch
import numpy as np
import sys
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score, f1_score, matthews_corrcoef

class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


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

def computeAUROC(dataGT, dataPRED, classCount=14):
    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC

def get_classwise_mean_std(data):
    data = np.array(data)
    class_wise_mean, class_wise_std = [],[]
    _, n_class = data.shape
    for ic in range(n_class):
        class_wise_mean.append(np.mean(data[:,ic]))
        class_wise_std.append(np.std(data[:,ic]))
    return [class_wise_mean, class_wise_std]

def meanMCC(ground_truth, predictions):
    mcc_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            fpr, tpr, thresholds = roc_curve(ground_truth[:, i], predictions[:, i])
            youden_j = tpr - fpr
            optimal_threshold = thresholds[np.argmax(youden_j)]
            binary_predictions = (predictions[:, i] > optimal_threshold).astype(int)
            mcc = matthews_corrcoef(ground_truth[:, i], binary_predictions)
            mcc_scores.append(mcc)

    mean_score = np.mean(mcc_scores)
    return mean_score, mcc_scores

def meanAP(ground_truth, predictions):
    # Compute mean Average Precision (mAP)
    ap_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            ap = average_precision_score(ground_truth[:, i], predictions[:, i])
            ap_scores.append(ap)

    mean_score = np.mean(ap_scores)
    return mean_score, ap_scores

def meanAUC(ground_truth, predictions):
    # Compute mean Area Under the ROC Curve (mAUC)
    auc_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            auc = roc_auc_score(ground_truth[:, i], predictions[:, i])
            auc_scores.append(auc)

    mean_score = np.mean(auc_scores)
    return mean_score, auc_scores
def meanF1(ground_truth, predictions):
    # Compute mean F1 score (mF1)
    f1_scores = []
    if ground_truth.shape[1] == 1:
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
        youden_j = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_j)]
        binary_predictions = (predictions > optimal_threshold).astype(int)
        f1 = f1_score(ground_truth, binary_predictions)
        f1_scores.append(f1)
    else:
        for i in range(ground_truth.shape[1]):
            if np.any(ground_truth[:, i]):
                fpr, tpr, thresholds = roc_curve(ground_truth[:, i], predictions[:, i])
                youden_j = tpr - fpr
                optimal_threshold = thresholds[np.argmax(youden_j)]
                binary_predictions = (predictions[:, i] > optimal_threshold).astype(int)
                f1 = f1_score(ground_truth[:, i], binary_predictions)
                f1_scores.append(f1)

    mean_score = np.mean(f1_scores)
    return mean_score, f1_scores

def accuracy(ground_truth, predictions):
    # Binary classification
    if ground_truth.shape[1] == 1:
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
        youden_j = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_j)]
        predictions = (predictions > optimal_threshold).astype(int)
    else:
        ground_truth = np.argmax(ground_truth, axis = 1)
        predictions = np.argmax(predictions, axis = 1)
    
    return accuracy_score(ground_truth, predictions)

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()

#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         #print(target, pred, batch_size, correct, res)
#         return res

def save_model(model, optimizer, log_writter, epoch, save_file):
    print('==> Saving...',file=log_writter)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_popar_weight(model, ckpt, log_writter):
    for k, v in ckpt["model"].items():
        k1 = k.replace("module.swin_model", "swinViT")
        k1 = k1.replace("layers.0", "layers1.0")
        k1 = k1.replace("layers.1", "layers2.0")
        k1 = k1.replace("layers.2", "layers3.0")
        k1 = k1.replace("layers.3", "layers4.0")
        k1 = k1.replace(".fc", ".linear")
        if k1 in model.state_dict().keys():
            model.state_dict()[k1].copy_(ckpt["model"][k])
        else:
            print(k," --> ", k1," miss matched",  file=log_writter)

    return model

def dice_score(im1, im2, empty_score=1.0):
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
        sum += dice_score(y_true[i,:,:,:],y_pred[i,:,:,:])
    return sum/y_true.shape[0]

def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))


def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        lr = init_lr * (1 - epoch / max_epoch)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr

def step_decay(step,conf):

    lr = conf.lr
    progress = (step - 20) / float(conf.epochs - 20)
    progress = np.clip(progress, 0.0, 1.0)
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))

    lr = lr * np.minimum(1., step / 20)

    return lr


def load_swin_pretrained(ckpt, model, writter=sys.stdout):
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
            print(f"Error in loading {k}, passing......", file=writter)
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
    print(msg, file=writter)

    del ckpt
    torch.cuda.empty_cache()