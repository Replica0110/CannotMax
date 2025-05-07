import json
import os
import shutil
import time

import torch
from loguru import logger


def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, train_acc, val_acc, config, save_path, is_best=False, best_save_path=None):
    """
    保存模型检查点及其相关状态。

    参数:
        epoch (int): 当前的 epoch 数。
        model (torch.nn.Module): 要保存的模型。
        optimizer (torch.optim.Optimizer): 优化器状态。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器状态。
        train_loss (float): 当前 epoch 的训练损失。
        val_loss (float): 当前 epoch 的验证损失。
        train_acc (float): 当前 epoch 的训练准确率。
        val_acc (float): 当前 epoch 的验证准确率。
        config (dict): 训练配置。
        save_path (str): 当前 epoch 模型保存的完整路径 (例如 'models/epoch_10')。
        is_best (bool): 当前模型是否是最佳模型。
        best_save_path (str): 最佳模型保存的完整路径 (例如 'models/best_epoch')。
    """
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, 'model.pth')
    optimizer_file = os.path.join(save_path, 'optimizer.pth')
    scheduler_file = os.path.join(save_path, 'scheduler.pth')
    state_file = os.path.join(save_path, 'training_state.json')

    # 保存模型权重
    torch.save(model, model_file)
    # 保存优化器状态
    torch.save(optimizer.state_dict(), optimizer_file)
    # 保存调度器状态 (如果存在)
    if scheduler:
        torch.save(scheduler.state_dict(), scheduler_file)

    # 构建 JSON 可序列化的训练状态
    # 注意：将所有数值类型转换为 Python 原生类型，以防它们是 numpy 类型
    training_state_json = {
        'epoch': int(epoch + 1), # epoch从0开始，所以保存时+1表示已完成的epoch
        'model_state_dict_path': 'model.pth', # 相对路径
        'optimizer_state_dict_path': 'optimizer.pth', # 相对路径
        'scheduler_state_dict_path': 'scheduler.pth' if scheduler else None, # 相对路径
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'config': config, # 训练配置通常是JSON兼容的
        'timestamp': float(time.time())
    }

    with open(state_file, 'w') as f:
        json.dump(training_state_json, f, indent=4, ensure_ascii=False)

    logger.info(f"检查点已保存至: {save_path}")

    if is_best and best_save_path:
        os.makedirs(best_save_path, exist_ok=True)
        best_model_file = os.path.join(best_save_path, 'model.pth')
        best_optimizer_file = os.path.join(best_save_path, 'optimizer.pth')
        best_scheduler_file = os.path.join(best_save_path, 'scheduler.pth')
        best_state_file = os.path.join(best_save_path, 'training_state.json')

        shutil.copyfile(model_file, best_model_file)
        shutil.copyfile(optimizer_file, best_optimizer_file)
        if scheduler and os.path.exists(scheduler_file):
            shutil.copyfile(scheduler_file, best_scheduler_file)
        # 最佳状态文件内容与当前epoch的JSON状态一致
        with open(best_state_file, 'w') as f:
            json.dump(training_state_json, f, indent=4, ensure_ascii=False)
        logger.info(f"最佳模型已更新并保存至: {best_save_path}")


# 辅助函数：加载检查点
def load_checkpoint(checkpoint_dir, model, optimizer=None, scheduler=None, device='cpu'):
    """
    加载模型检查点。

    参数:
        checkpoint_dir (str): 包含模型和状态文件的文件夹路径。
        model (torch.nn.Module): 要加载权重的模型。
        optimizer (torch.optim.Optimizer, optional): 要加载状态的优化器。
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 要加载状态的调度器。
        device (str or torch.device): 模型将加载到的设备。

    返回:
        tuple: (start_epoch, config, train_loss, val_loss, train_acc, val_acc)
               如果加载失败则返回 (0, None, float('inf'), float('inf'), 0, 0)
    """
    state_file = os.path.join(checkpoint_dir, 'training_state.json')

    if not os.path.exists(state_file):
        logger.warning(f"状态文件 'training_state.json' 未在 '{checkpoint_dir}' 中找到。将从头开始训练。")
        return 0, None, float('inf'), float('inf'), 0.0, 0.0

    try:
        logger.info(f"从 '{checkpoint_dir}' 加载检查点...")

        # 加载 JSON 训练状态
        with open(state_file, 'r') as f:
            training_state_json = json.load(f)

        model_file = os.path.join(checkpoint_dir, training_state_json['model_state_dict_path'])
        optimizer_file = os.path.join(checkpoint_dir, training_state_json['optimizer_state_dict_path'])
        scheduler_file_path = training_state_json.get('scheduler_state_dict_path')
        if scheduler_file_path:
            scheduler_file = os.path.join(checkpoint_dir, scheduler_file_path)
        else:
            scheduler_file = None


        if not os.path.exists(model_file):
            logger.error(f"模型文件 '{model_file}' 未找到。加载失败。")
            return 0, None, float('inf'), float('inf'), 0.0, 0.0

        # 加载模型
        model = torch.load(model_file, map_location=device)
        model.to(device)

        if optimizer and os.path.exists(optimizer_file):
            optimizer.load_state_dict(torch.load(optimizer_file, map_location=device))
            # 将优化器状态中的张量也移动到正确的设备
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        elif optimizer:
            logger.warning(f"优化器文件 '{optimizer_file}' 未找到。优化器状态未加载。")


        if scheduler and scheduler_file and os.path.exists(scheduler_file):
            scheduler.load_state_dict(torch.load(scheduler_file, map_location=device))
        elif scheduler and scheduler_file: # scheduler_file_path 存在但文件本身不存在
            logger.warning(f"调度器文件 '{scheduler_file}' 未找到。调度器状态未加载。")
        elif scheduler and not scheduler_file: # scheduler_file_path 在json中为None
            logger.info("JSON状态中未指定调度器文件路径。调度器状态未加载。")


        start_epoch = training_state_json.get('epoch', 0)
        loaded_config = training_state_json.get('config', None)
        train_loss = training_state_json.get('train_loss', float('inf'))
        val_loss = training_state_json.get('val_loss', float('inf'))
        train_acc = training_state_json.get('train_acc', 0.0)
        val_acc = training_state_json.get('val_acc', 0.0)

        logger.info(f"检查点加载成功。将从 epoch {start_epoch + 1} 继续训练。")
        logger.info(f"上一个 epoch ({start_epoch}) 的性能: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        return start_epoch, loaded_config, train_loss, val_loss, train_acc, val_acc

    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        logger.warning("将从头开始训练。")
        return 0, None, float('inf'), float('inf'), 0.0, 0.0


def manage_recent_checkpoints(base_save_dir, current_epoch_num, num_to_keep=3):
    """
    删除比最近 num_to_keep 个更早的 epoch 文件夹。
    """
    epoch_dirs = []
    for item in os.listdir(base_save_dir):
        if item.startswith("epoch_") and os.path.isdir(os.path.join(base_save_dir, item)):
            try:
                num = int(item.split("_")[1])
                epoch_dirs.append((num, os.path.join(base_save_dir, item)))
            except ValueError:
                continue # 忽略非标准 epoch 文件夹名

    # 按 epoch 号排序，最新的在前
    epoch_dirs.sort(key=lambda x: x[0], reverse=True)

    # 删除多余的旧文件夹
    if len(epoch_dirs) > num_to_keep:
        for _, dir_to_delete in epoch_dirs[num_to_keep:]:
            if os.path.exists(dir_to_delete):
                logger.info(f"删除旧的检查点文件夹: {dir_to_delete}")
                shutil.rmtree(dir_to_delete)