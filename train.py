import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import time
from torch.cuda.amp import GradScaler
from loguru import logger

from checkpoint import save_checkpoint, load_checkpoint, manage_recent_checkpoints

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os
import numpy as np
import pandas as pd # 仍然需要pandas进行后续操作和类型兼容
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import time
from torch.cuda.amp import GradScaler
from loguru import logger

# 导入 pyarrow.csv
import pyarrow.csv as pv
import pyarrow as pa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data(csv_file):
    """预处理CSV文件，使用 PyArrow 读取，将异常值修正为合理范围"""
    logger.info(f"使用 PyArrow 预处理数据文件: {csv_file}")

    # PyArrow 读取 CSV
    read_options = pv.ReadOptions()
    read_options.skip_rows = 1  # 跳过第一行（原始文件的表头）
    read_options.autogenerate_column_names = True

    parse_options = pv.ParseOptions()
    convert_options = pv.ConvertOptions()
    try:
        # 使用 pyarrow.csv.read_csv 读取
        table = pv.read_csv(csv_file,
                            read_options=read_options,
                            parse_options=parse_options,
                            convert_options=convert_options)

        data = table.to_pandas()

    except pa.ArrowInvalid as e:
        logger.error(f"PyArrow 读取CSV时出错: {e}")
        logger.warning("尝试回退到 Pandas 读取...")
        try:
            # 回退到Pandas读取，并尝试处理坏行
            data = pd.read_csv(csv_file, header=None, skiprows=1, on_bad_lines='warn')
            if data.empty and os.path.getsize(csv_file) > 0: # 检查文件是否有内容
                 logger.error("DataFrame 为空，但文件非空。请检查CSV文件。")
                 return 0
        except Exception as pd_e:
            logger.error(f"Pandas 读取失败: {pd_e}")
            return 0

    if data.empty:
        logger.error("读取数据后DataFrame为空，请检查CSV文件或读取参数。")
        return 0


    logger.info(f"原始数据形状: {data.shape}")

    try:
        features = data.iloc[:, :-1]
        labels = data.iloc[:, -1]
    except IndexError as e:
        logger.error(f"从读取的数据中提取特征和标签时出错: {e}")
        logger.error("可能是因为CSV列数不符合预期，或者文件未能正确读取。")
        return 0 # 指示预处理失败


    # 统计极端值
    try:
        numeric_features = features.apply(pd.to_numeric, errors='coerce')
        extreme_values = (np.abs(numeric_features) > 20).sum().sum()
        if extreme_values > 0:
            logger.warning(f"发现 {extreme_values} 个绝对值大于20的特征值")

        feature_min = numeric_features.min().min()
        feature_max = numeric_features.max().max()
        feature_mean = numeric_features.mean().mean()
        feature_std = numeric_features.std().mean()
    except Exception as e:
        logger.error(f"统计特征值时发生错误: {e}")
        logger.warning("这可能表明特征列中存在非数值数据，或数据为空。")
        feature_min, feature_max, feature_mean, feature_std = np.nan, np.nan, np.nan, np.nan


    # 检查标签
    invalid_labels = labels.apply(lambda x: x not in ["L", "R"]).sum()
    if invalid_labels > 0:
        logger.warning(f"发现 {invalid_labels} 个无效标签")


    logger.info(f"特征值范围: [{feature_min}, {feature_max}]")
    logger.info(f"特征值平均值: {feature_mean:.4f}, 标准差: {feature_std:.4f}")

    return data.shape[1]


class ArknightsDataset(Dataset):
    def __init__(self, csv_file, max_value=None):
        data = pd.read_csv(csv_file, header=None, skiprows=1)
        features = data.iloc[:, :-1].values.astype(np.float32)
        labels = data.iloc[:, -1].map({"L": 0, "R": 1}).values
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)

        # 分割双方单位
        feature_count = features.shape[1]
        midpoint = feature_count // 2
        left_counts = np.abs(features[:, :midpoint])
        right_counts = np.abs(features[:, midpoint:])
        left_signs = np.sign(features[:, :midpoint])
        right_signs = np.sign(features[:, midpoint:])

        if max_value is not None:
            left_counts = np.clip(left_counts, 0, max_value)
            right_counts = np.clip(right_counts, 0, max_value)

        # 转换为 PyTorch 张量，并一次性加载到 GPU
        self.left_signs = torch.from_numpy(left_signs).to(device)
        self.right_signs = torch.from_numpy(right_signs).to(device)
        self.left_counts = torch.from_numpy(left_counts).to(device)
        self.right_counts = torch.from_numpy(right_counts).to(device)
        self.labels = torch.from_numpy(labels).float().to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.left_signs[idx],
            self.left_counts[idx],
            self.right_signs[idx],
            self.right_counts[idx],
            self.labels[idx],
        )


class UnitAwareTransformer(nn.Module):
    def __init__(self, num_units, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 嵌入层
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # 注意力层与FFN
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()

        for _ in range(num_layers):
            # 敌方注意力层
            self.enemy_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=0.2
                )
            )
            self.enemy_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            # 友方注意力层
            self.friend_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=0.2
                )
            )
            self.friend_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            # 初始化注意力层参数
            nn.init.xavier_uniform_(self.enemy_attentions[-1].in_proj_weight)
            nn.init.xavier_uniform_(self.friend_attentions[-1].in_proj_weight)

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1)
        )

    def forward(self, left_sign, left_count, right_sign, right_count):
        # 提取Top3兵种特征
        left_values, left_indices = torch.topk(left_count, k=3, dim=1)
        right_values, right_indices = torch.topk(right_count, k=3, dim=1)

        # 嵌入
        left_feat = self.unit_embed(left_indices)  # (B, 3, 128)
        right_feat = self.unit_embed(right_indices)  # (B, 3, 128)

        embed_dim = self.embed_dim

        # 前x维不变，后y维 *= 数量，但使用缩放后的值
        left_feat = torch.cat(
            [
                left_feat[..., : embed_dim // 2],  # 前x维
                left_feat[..., embed_dim // 2 :]
                * left_values.unsqueeze(-1),  # 后y维乘数量
            ],
            dim=-1,
        )
        right_feat = torch.cat(
            [
                right_feat[..., : embed_dim // 2],
                right_feat[..., embed_dim // 2 :] * right_values.unsqueeze(-1),
            ],
            dim=-1,
        )

        # FFN
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        # 生成mask (B, 3) 0.1防一手可能的浮点误差
        left_mask = left_values > 0.1
        right_mask = right_values > 0.1

        for i in range(self.num_layers):
            # 敌方注意力
            delta_left, _ = self.enemy_attentions[i](
                query=left_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )
            delta_right, _ = self.enemy_attentions[i](
                query=right_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)

            # 友方注意力
            delta_left, _ = self.friend_attentions[i](
                query=left_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )
            delta_right, _ = self.friend_attentions[i](
                query=right_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)

        # 输出战斗力
        L = self.fc(left_feat).squeeze(-1) * left_mask
        R = self.fc(right_feat).squeeze(-1) * right_mask

        # 计算战斗力差输出概率，'L': 0, 'R': 1，R大于L时输出大于0.5
        output = torch.sigmoid(R.sum(1) - L.sum(1))

        return output


def train_one_epoch(model, train_loader, criterion, optimizer, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for ls, lc, rs, rc, labels in train_loader:
        ls, lc, rs, rc, labels = [x.to(device, non_blocking=True) for x in (ls, lc, rs, rc, labels)]

        optimizer.zero_grad()

        # 检查输入值范围
        if (
            torch.isnan(ls).any()
            or torch.isnan(lc).any()
            or torch.isnan(rs).any()
            or torch.isnan(rc).any()
        ):
            logger.warning("警告：输入数据包含NaN，跳过该批次")
            continue

        if (
            torch.isinf(ls).any()
            or torch.isinf(lc).any()
            or torch.isinf(rs).any()
            or torch.isinf(rc).any()
        ):
            logger.warning("警告：输入数据包含Inf，跳过该批次")
            continue

        # 确保labels严格在0-1之间
        if (labels < 0).any() or (labels > 1).any():
            logger.warning("警告：标签值不在[0,1]范围内，进行修正")
            labels = torch.clamp(labels, 0, 1)

        try:
            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                outputs = model(ls, lc, rs, rc).squeeze()
                # 确保输出在合理范围内
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.warning("警告：模型输出包含NaN或Inf，跳过该批次")
                    continue

                # 确保输出严格在0-1之间，因为BCELoss需要
                if (outputs < 0).any() or (outputs > 1).any():
                    logger.warning("警告：模型输出不在[0,1]范围内，进行修正")
                    outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

                loss = criterion(outputs, labels)

            # 检查loss是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"警告：损失值为 {loss.item()}, 跳过该批次")
                continue

            if scaler:  # 使用混合精度
                scaler.scale(loss).backward()
                # 梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # 不使用混合精度
                loss.backward()
                # 梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        except RuntimeError as e:
            logger.warning(f"警告：训练过程中出错 - {str(e)}")
            continue

    return total_loss / max(1, len(train_loader)), 100 * correct / max(1, total)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for ls, lc, rs, rc, labels in data_loader:
            ls, lc, rs, rc, labels = [x.to(device, non_blocking=True) for x in (ls, lc, rs, rc, labels)]

            # 检查输入值范围
            if (
                torch.isnan(ls).any()
                or torch.isnan(lc).any()
                or torch.isnan(rs).any()
                or torch.isnan(rc).any()
                or torch.isinf(ls).any()
                or torch.isinf(lc).any()
                or torch.isinf(rs).any()
                or torch.isinf(rc).any()
            ):
                logger.warning("警告：评估时输入数据包含NaN或Inf，跳过该批次")
                continue

            # 确保labels严格在0-1之间
            if (labels < 0).any() or (labels > 1).any():
                labels = torch.clamp(labels, 0, 1)

            try:
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(ls, lc, rs, rc).squeeze()
                    # 确保输出在合理范围内
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        logger.warning("警告：评估时模型输出包含NaN或Inf，跳过该批次")
                        continue

                    # 确保输出严格在0-1之间，因为BCELoss需要
                    if (outputs < 0).any() or (outputs > 1).any():
                        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

                loss = criterion(outputs, labels)

                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            except RuntimeError as e:
                logger.warning(f"警告：评估过程中出错 - {str(e)}")
                continue

    return total_loss / max(1, len(data_loader)), 100 * correct / max(1, total)


def stratified_random_split(dataset, test_size=0.1, seed=42):
    labels = dataset.labels  # 假设 labels 是一个 GPU tensor
    if device != "cpu":
        labels = labels.cpu()  # 移动到 CPU 上进行操作
    labels = labels.numpy()  # 转换为 numpy array

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
    )


def main():
    # 配置参数
    config = {
        "data_file": "arknights.csv",
        "batch_size": 2048,
        "test_size": 0.1,
        "embed_dim": 256,
        "n_layers": 4,
        "num_heads": 8,
        "lr": 5e-4,
        "epochs": 100,
        "seed": 42,
        "save_dir": "models",
        "max_feature_value": 100,
        "num_workers": 0 if torch.cuda.is_available() else 0,
        "num_recent_checkpoints_to_keep": 3,
        "best_metric": "val_acc",  # 'val_acc' 或 'val_loss'
    }

    # 创建保存目录
    os.makedirs(config["save_dir"], exist_ok=True)
    best_epoch_dir = os.path.join(config["save_dir"], "best_epoch")

    # 设置随机种子
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 初始化 GradScaler 用于混合精度训练
    scaler = None
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            scaler = GradScaler() # 如果是老版本
        logger.info("CUDA可用，已启用混合精度训练的GradScaler。")

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logger.warning("警告：未检测到GPU，将在CPU上运行训练!")

    # --- 数据加载 ---
    num_total_cols = preprocess_data(config["data_file"])
    if num_total_cols == 0:
        logger.error("数据预处理失败或未返回有效列数，程序终止。")
        return

    num_units_per_side = (num_total_cols - 1) // 2
    dataset = ArknightsDataset(config["data_file"], max_value=config["max_feature_value"])
    train_dataset, val_dataset = stratified_random_split(dataset, test_size=config["test_size"], seed=config["seed"])
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])

    # --- 初始化模型、优化器、损失函数、调度器 ---
    model = UnitAwareTransformer(
        num_units=num_units_per_side,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["n_layers"],
    ).to(device)
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # --- 默认尝试恢复 ---
    start_epoch = 0
    best_val_metric_value = float('-inf') if config["best_metric"] == "val_acc" else float('inf')

    checkpoint_to_load = None
    if os.path.exists(best_epoch_dir):
        checkpoint_to_load = best_epoch_dir
    else:
        # 查找最新的 epoch_X 文件夹
        epoch_dirs = []
        for item in os.listdir(config["save_dir"]):
            if item.startswith("epoch_") and os.path.isdir(os.path.join(config["save_dir"], item)):
                try:
                    num = int(item.split("_")[1])
                    epoch_dirs.append((num, os.path.join(config["save_dir"], item)))
                except ValueError:
                    continue
        if epoch_dirs:
            epoch_dirs.sort(key=lambda x: x[0], reverse=True)
            checkpoint_to_load = epoch_dirs[0][1]

    if checkpoint_to_load and os.path.isdir(checkpoint_to_load):
        loaded_epoch, loaded_config, _, loaded_val_loss, _, loaded_val_acc = load_checkpoint(
            checkpoint_to_load, model, optimizer, scheduler, device
        )
        if loaded_epoch > 0:
            start_epoch = loaded_epoch
            logger.info(f"成功从 {checkpoint_to_load} 恢复，第{start_epoch}轮")
            if config["best_metric"] == "val_acc":
                best_val_metric_value = loaded_val_acc
            else:
                best_val_metric_value = loaded_val_loss
        else:
            logger.info("检查点加载失败，将从头开始训练。")
    else:
        logger.info("未找到有效检查点，从头开始训练。")

    # --- 训练循环 ---
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    total_training_start_time = time.time()

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # 更新学习率
        scheduler.step()

        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 保存模型
        current_epoch_save_dir = os.path.join(config["save_dir"], f"epoch_{epoch + 1}")

        is_best = False
        if config["best_metric"] == "val_acc":
            if val_acc > best_val_metric_value:
                best_val_metric_value = val_acc
                is_best = True
        else:
            if val_loss < best_val_metric_value:
                best_val_metric_value = val_loss
                is_best = True

        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            config=config,
            save_path=current_epoch_save_dir,
            is_best=is_best,
            best_save_path=best_epoch_dir if is_best else None
        )

        manage_recent_checkpoints(config["save_dir"], epoch + 1, config["num_recent_checkpoints_to_keep"])

        # 日志
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        if is_best:
            logger.info(f"新的最佳模型已保存! ({config['best_metric']}: {best_val_metric_value:.4f})")

        epoch_duration = time.time() - epoch_start_time
        elapsed_time = time.time() - total_training_start_time
        avg_epoch_time = elapsed_time / (epoch - start_epoch + 1)
        estimated_total_time = avg_epoch_time * (config["epochs"] - start_epoch)
        remaining_time = estimated_total_time - elapsed_time

        logger.info(f"耗时：{elapsed_time / 60:.2f}分钟 | 剩余预估时间：{remaining_time / 60:.2f}分钟 | 单轮训练耗时：{epoch_duration:.2f}秒")

        logger.info("-" * 50)

    logger.info(f"训练完成! 最终最佳验证指标 ({config['best_metric']}): {best_val_metric_value:.4f}")
    logger.info(f"最佳模型保存在: {best_epoch_dir}")


if __name__ == "__main__":
    main()
