# Derived from EGE-UNet (https://github.com/JCruan519/EGE-UNet),
# licensed under Apache-2.0. Modified to add fixed-threshold metrics and HD95.
# See NOTICE.

import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
from utils import save_imgs

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None


def _fixed_threshold(config):
    return float(getattr(config, 'threshold', 0.5))


def _surface(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask

    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    eroded = padded[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            eroded &= padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    return mask & ~eroded


def _distances_to_surface(source_surface, target_surface):
    source_points = np.argwhere(source_surface)
    target_points = np.argwhere(target_surface)
    if source_points.size == 0 or target_points.size == 0:
        return np.array([], dtype=np.float32)

    if cv2 is not None:
        distance_input = (~target_surface).astype(np.uint8)
        distance_map = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
        return distance_map[source_surface].astype(np.float32)

    if distance_transform_edt is not None:
        distance_map = distance_transform_edt(~target_surface)
        return distance_map[source_surface].astype(np.float32)

    distances = []
    target_points = target_points.astype(np.float32)
    for start in range(0, len(source_points), 1024):
        chunk = source_points[start:start + 1024].astype(np.float32)
        diff = chunk[:, None, :] - target_points[None, :, :]
        distances.append(np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1))
    return np.concatenate(distances).astype(np.float32)


def _hd95(pred_mask, gt_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    if not pred_mask.any() and not gt_mask.any():
        return 0.0
    if pred_mask.any() != gt_mask.any():
        return float(np.hypot(*pred_mask.shape))

    pred_surface = _surface(pred_mask)
    gt_surface = _surface(gt_mask)
    pred_to_gt = _distances_to_surface(pred_surface, gt_surface)
    gt_to_pred = _distances_to_surface(gt_surface, pred_surface)
    if pred_to_gt.size == 0 and gt_to_pred.size == 0:
        return 0.0
    distances = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.percentile(distances, 95))


def _mean_hd95(y_pre, y_true):
    y_pre = np.asarray(y_pre, dtype=bool)
    y_true = np.asarray(y_true, dtype=bool)
    if y_pre.ndim == 2:
        y_pre = y_pre[None, ...]
        y_true = y_true[None, ...]
    return float(np.mean([_hd95(pred, gt) for pred, gt in zip(y_pre, y_true)]))


def _binary_metrics(preds, gts, threshold):
    pred_masks = np.asarray(preds) >= threshold
    gt_masks = np.asarray(gts) >= 0.5
    y_pre = pred_masks.reshape(-1).astype(np.uint8)
    y_true = gt_masks.reshape(-1).astype(np.uint8)

    confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    hd95 = _mean_hd95(pred_masks, gt_masks)

    return {
        'threshold': threshold,
        'miou': miou,
        'dsc': f1_or_dsc,
        'hd95': hd95,
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion': confusion,
    }


def _segmentation_output(model_output):
    if isinstance(model_output, tuple):
        model_output = model_output[-1]
    if isinstance(model_output, tuple):
        model_output = model_output[0]
    return model_output


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        gt_pre, out = model(images)
        loss = criterion(gt_pre, out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, out = model(img)
            loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            out = _segmentation_output(out)
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    val_loss = np.mean(loss_list)
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    threshold = _fixed_threshold(config)
    metrics = _binary_metrics(preds, gts, threshold)
    accuracy = metrics['accuracy']
    sensitivity = metrics['sensitivity']
    specificity = metrics['specificity']
    dsc = metrics['dsc']
    miou = metrics['miou']
    hd95 = metrics['hd95']
    confusion = metrics['confusion']

    if epoch % config.val_interval == 0:
        log_info = (
            f'val epoch: {epoch}, threshold: {threshold:.2f}, loss: {val_loss:.4f}, '
            f'mIoU: {miou:.6f}, DSC: {dsc:.6f}, HD95: {hd95:.6f}, '
            f'accuracy: {accuracy:.6f}, sensitivity: {sensitivity:.6f}, '
            f'specificity: {specificity:.6f}, confusion_matrix: {confusion}'
        )
    else:
        log_info = (
            f'val epoch: {epoch}, threshold: {threshold:.2f}, loss: {val_loss:.4f}, '
            f'mIoU: {miou:.6f}, DSC: {dsc:.6f}, HD95: {hd95:.6f}'
        )
    print(log_info)
    logger.info(log_info)
    
    return {
        'loss': val_loss,
        'miou': miou,
        'dsc': dsc,
        'hd95': hd95,
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'threshold': threshold,
    }


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, out = model(img)
            loss = criterion(gt_pre, out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            out = _segmentation_output(out)
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        threshold = _fixed_threshold(config)
        metrics = _binary_metrics(preds, gts, threshold)
        accuracy = metrics['accuracy']
        sensitivity = metrics['sensitivity']
        specificity = metrics['specificity']
        dsc = metrics['dsc']
        miou = metrics['miou']
        hd95 = metrics['hd95']
        confusion = metrics['confusion']

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = (
            f'test: threshold: {threshold:.2f}, loss: {np.mean(loss_list):.4f}, '
            f'mIoU: {miou:.6f}, DSC: {dsc:.6f}, HD95: {hd95:.6f}, '
            f'accuracy: {accuracy:.6f}, sensitivity: {sensitivity:.6f}, '
            f'specificity: {specificity:.6f}, confusion_matrix: {confusion}'
        )
        print(log_info)
        logger.info(log_info)

    return {
        'loss': float(np.mean(loss_list)),
        'metrics': metrics,
    }
