# Derived from EGE-UNet (https://github.com/JCruan519/EGE-UNet),
# licensed under Apache-2.0. Modified for HLLK/BRR training and reporting.
# See NOTICE.

import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.hbr_unet import HBRUNet

from engine import *
import os
import sys
import csv

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


RESULTS_SUMMARY_COLUMNS = [
    'dataset',
    'seed',
    'modules',
    'threshold',
    'best_epoch',
    'mIoU',
    'DSC',
    'sensitivity',
    'specificity',
    'HD95',
    'params',
    'result_dir',
]


def _append_results_summary(config, active_modules, best_epoch, test_metrics, params, logger):
    metrics = test_metrics.get('metrics', {})
    project_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = getattr(config, 'results_summary_path', None)
    if summary_path is None:
        summary_path = os.path.join(project_dir, 'results_summary.csv')
    elif not os.path.isabs(summary_path):
        summary_path = os.path.join(project_dir, summary_path)
    result_dir = os.path.normpath(config.work_dir)

    existing_rows = []
    if os.path.exists(summary_path):
        with open(summary_path, 'r', newline='', encoding='utf-8') as summary_file:
            reader = csv.DictReader(summary_file)
            existing_rows = list(reader)
            for old_row in existing_rows:
                if old_row.get('result_dir') == result_dir:
                    logger.info(f'results summary already contains result_dir: {result_dir}')
                    return

        if reader.fieldnames != RESULTS_SUMMARY_COLUMNS:
            with open(summary_path, 'w', newline='', encoding='utf-8') as summary_file:
                writer = csv.DictWriter(summary_file, fieldnames=RESULTS_SUMMARY_COLUMNS)
                writer.writeheader()
                for old_row in existing_rows:
                    normalized_row = {column: old_row.get(column, '') for column in RESULTS_SUMMARY_COLUMNS}
                    writer.writerow(normalized_row)

    row = {
        'dataset': config.datasets,
        'seed': getattr(config, 'seed', ''),
        'modules': '+'.join(active_modules),
        'threshold': f"{float(metrics.get('threshold', getattr(config, 'threshold', 0.5))):.2f}",
        'best_epoch': best_epoch,
        'mIoU': metrics.get('miou', ''),
        'DSC': metrics.get('dsc', ''),
        'sensitivity': metrics.get('sensitivity', ''),
        'specificity': metrics.get('specificity', ''),
        'HD95': metrics.get('hd95', ''),
        'params': params,
        'result_dir': result_dir,
    }

    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    write_header = not os.path.exists(summary_path)
    with open(summary_path, 'a', newline='', encoding='utf-8') as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=RESULTS_SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f'append results summary: {summary_path}')


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network in ('hbr_unet', 'egeunet'):
        model = HBRUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'], 
                        c_list=model_cfg['c_list'], 
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'],
                        use_high_level_large_kernel=model_cfg.get('use_high_level_large_kernel', False),
                        use_decoder_brr=model_cfg.get('use_decoder_brr', False),
                        hllk_kernel_size=model_cfg.get('hllk_kernel_size', 7),
                        use_brr_region_branch=model_cfg.get('use_brr_region_branch', True),
                        use_brr_boundary_branch=model_cfg.get('use_brr_boundary_branch', True),
                        )
    else: raise Exception('network in not right!')
    model = model.cuda()
    model_params = sum(p.numel() for p in model.parameters())
    active_modules = []
    active_modules.append('encoder_stage3_conv3x3')
    active_modules.append('encoder_stage2_conv3x3')
    if model_cfg.get('use_high_level_large_kernel', False):
        active_modules.append(f"high_level_large_kernel_k{model_cfg.get('hllk_kernel_size', 7)}_encoder4_encoder5_encoder6")
    else:
        active_modules.append('original_high_level_ghpa_encoder4_encoder5_encoder6')
    if model_cfg['bridge']:
        active_modules.append('original_gab1_gab2_gab3_gab4_gab5')
    if model_cfg.get('use_decoder_brr', False):
        enabled_brr_branches = []
        if model_cfg.get('use_brr_region_branch', True):
            enabled_brr_branches.append('region')
        if model_cfg.get('use_brr_boundary_branch', True):
            enabled_brr_branches.append('boundary')
        active_modules.append(
            f"brr_{'_'.join(enabled_brr_branches)}_after_gab3_gab2_gab1"
        )
    active_modules.append('bilinear_decoder_upsampling')
    if model_cfg['gt_ds']:
        active_modules.append('gt_deep_supervision')
    gab_log_info = (
        f"model switches: experiment_tag={getattr(config, 'experiment_tag', '') or 'none'}, "
        f"env_HBR_USE_HLLK={os.environ.get('HBR_USE_HLLK', os.environ.get('EGE_USE_HLLK', 'default'))}, "
        f"env_HBR_USE_BRR={os.environ.get('HBR_USE_BRR', os.environ.get('EGE_USE_BRR', 'default'))}, "
        f"env_HBR_HLLK_KERNEL_SIZE={os.environ.get('HBR_HLLK_KERNEL_SIZE', os.environ.get('EGE_HLLK_KERNEL_SIZE', 'default'))}, "
        f"env_HBR_BRR_USE_REGION={os.environ.get('HBR_BRR_USE_REGION', os.environ.get('EGE_BRR_USE_REGION', 'default'))}, "
        f"env_HBR_BRR_USE_BOUNDARY={os.environ.get('HBR_BRR_USE_BOUNDARY', os.environ.get('EGE_BRR_USE_BOUNDARY', 'default'))}, "
        f"bridge={model_cfg['bridge']}, "
        f"gt_ds={model_cfg['gt_ds']}, "
        f"use_high_level_large_kernel={model_cfg.get('use_high_level_large_kernel', False)}, "
        f"hllk_kernel_size={model_cfg.get('hllk_kernel_size', 7)}, "
        f"use_decoder_brr={model_cfg.get('use_decoder_brr', False)}, "
        f"use_brr_region_branch={model_cfg.get('use_brr_region_branch', True)}, "
        f"use_brr_boundary_branch={model_cfg.get('use_brr_boundary_branch', True)}, "
        f"threshold={getattr(config, 'threshold', 0.5)}, "
        f"num_workers={getattr(config, 'num_workers', 'unknown')}, "
        f"early_stop_patience={getattr(config, 'early_stop_patience', 'unknown')}, "
        f"active_modules={','.join(active_modules)}"
    )
    print(gab_log_info)
    logger.info(gab_log_info)
    writer.add_text('model_switches', gab_log_info, global_step=0)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    save_best_metric = getattr(config, 'save_best_metric', 'dsc').lower()
    assert save_best_metric in ['dsc', 'miou'], 'save_best_metric must be dsc or miou!'
    early_stop = getattr(config, 'early_stop', False)
    early_stop_patience = getattr(config, 'early_stop_patience', 60)
    early_stop_monitor = getattr(config, 'early_stop_monitor', 'val_dsc').lower()
    early_stop_mode = getattr(config, 'early_stop_mode', 'max').lower()
    early_stop_min_delta = getattr(config, 'early_stop_min_delta', 0.0)
    assert early_stop_monitor in ['val_dsc', 'val_miou', 'val_loss'], 'early_stop_monitor must be val_dsc, val_miou, or val_loss!'
    assert early_stop_mode in ['max', 'min'], 'early_stop_mode must be max or min!'
    best_score = -1
    start_epoch = 1
    best_epoch = 1
    no_improve_epochs = 0





    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        best_score = checkpoint.get('best_score', checkpoint.get('best_dsc', -1))
        best_epoch = checkpoint.get('best_epoch', checkpoint.get('min_epoch', 1))
        no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
        loss = checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, best_{save_best_metric}: {best_score:.4f}, best_epoch: {best_epoch}, no_improve_epochs: {no_improve_epochs}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        val_metrics = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )
        loss = val_metrics['loss']
        val_dsc = val_metrics['dsc']
        val_miou = val_metrics['miou']
        val_hd95 = val_metrics['hd95']
        val_threshold = val_metrics.get('threshold', config.threshold)
        monitor_scores = {
            'val_dsc': val_dsc,
            'val_miou': val_miou,
            'val_loss': loss,
        }
        monitor_score = monitor_scores[early_stop_monitor]
        writer.add_scalar('val/loss', loss, global_step=epoch)
        writer.add_scalar('val/dsc', val_dsc, global_step=epoch)
        writer.add_scalar('val/miou', val_miou, global_step=epoch)
        writer.add_scalar('val/hd95', val_hd95, global_step=epoch)
        writer.add_scalar('val/threshold', val_threshold, global_step=epoch)

        if best_score < 0:
            improved = True
        elif early_stop_mode == 'max':
            improved = monitor_score > best_score + early_stop_min_delta
        else:
            improved = monitor_score < best_score - early_stop_min_delta

        if improved:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            best_score = monitor_score
            best_epoch = epoch
            no_improve_epochs = 0
            log_info = f'save best model at epoch {epoch}, threshold: {val_threshold:.2f}, best_{early_stop_monitor}: {best_score:.4f}, val_loss: {loss:.4f}'
            print(log_info)
            logger.info(log_info)
        else:
            no_improve_epochs += 1
            if early_stop:
                log_info = (
                    f'early stop counter: {no_improve_epochs}/{early_stop_patience}, '
                    f'threshold: {val_threshold:.2f}, '
                    f'{early_stop_monitor}: {monitor_score:.4f}, '
                    f'best_{early_stop_monitor}: {best_score:.4f}'
                )
                print(log_info)
                logger.info(log_info)

        torch.save(
            {
                'epoch': epoch,
                'best_score': best_score,
                'best_epoch': best_epoch,
                'best_metric': early_stop_monitor,
                'no_improve_epochs': no_improve_epochs,
                'current_threshold': val_threshold,
                'current_dsc': val_dsc,
                'current_miou': val_miou,
                'current_hd95': val_hd95,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

        if early_stop and no_improve_epochs >= early_stop_patience:
            log_info = (
                f'early stopping at epoch {epoch}. '
                f'best_epoch: {best_epoch}, best_{early_stop_monitor}: {best_score:.4f}'
            )
            print(log_info)
            logger.info(log_info)
            break

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_metrics = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        _append_results_summary(config, active_modules, best_epoch, test_metrics, model_params, logger)
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-{save_best_metric}{best_score:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)
