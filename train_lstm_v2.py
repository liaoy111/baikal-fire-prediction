"""
第2步：LSTM火灾预测模型（修正版）
修改：随机划分数据集 + 类别权重
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os

# ============================================================
# ⚙️ 配置区
# ============================================================

DATA_FILE  = r"D:\PycharmProject\fire_project\data\processed_dataset.csv"
MODEL_DIR  = r"D:\PycharmProject\fire_project\output"
os.makedirs(MODEL_DIR, exist_ok=True)

SEQUENCE_LEN  = 7
BATCH_SIZE    = 16
EPOCHS        = 50
LEARNING_RATE = 0.001

FEATURE_COLS = ['temperature', 'dewpoint', 'wind_speed', 'ndvi']
LABEL_COL    = 'fire_label'

# ============================================================
# 数据集定义
# ============================================================

class FireSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(df, feature_cols, label_col, seq_len):
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[feature_cols].iloc[i-seq_len:i].values)
        y.append(df[label_col].iloc[i])
    return np.array(X), np.array(y)


# ============================================================
# LSTM模型
# ============================================================

class FireLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.classifier(last_step)


# ============================================================
# 训练和评估函数
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        total_loss += criterion(out, y).item() * len(y)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return (total_loss / len(loader.dataset),
            correct / len(loader.dataset),
            all_preds, all_labels)


# ============================================================
# 主流程
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

    # 1. 读取数据
    print("\n📂 读取数据...")
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"✅ 共 {len(df)} 天的数据")

    # 2. 特征归一化
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    # 3. 创建时间序列
    X, y = create_sequences(df, FEATURE_COLS, LABEL_COL, SEQUENCE_LEN)
    print(f"✅ 序列数据：X={X.shape}, y={y.shape}")

    # 4. 随机划分数据集（保证每个集合里火灾/无火灾比例均衡）
    idx = np.arange(len(X))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.3, random_state=42, stratify=y)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=42, stratify=y[idx_temp])

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]

    print(f"\n📊 数据划分：")
    print(f"   训练集：{len(X_train)} 个（火灾:{y_train.sum()} 无火灾:{len(y_train)-y_train.sum()}）")
    print(f"   验证集：{len(X_val)} 个（火灾:{y_val.sum()} 无火灾:{len(y_val)-y_val.sum()}）")
    print(f"   测试集：{len(X_test)} 个（火灾:{y_test.sum()} 无火灾:{len(y_test)-y_test.sum()}）")

    train_loader = DataLoader(FireSequenceDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(FireSequenceDataset(X_val, y_val),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(FireSequenceDataset(X_test, y_test),
                              batch_size=BATCH_SIZE)

    # 5. 初始化模型
    model = FireLSTM(input_size=len(FEATURE_COLS)).to(device)

    # 类别权重：降低火灾类别权重，防止模型偷懒只猜有火灾
    weight = torch.tensor([1.0, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)

    # 6. 训练循环
    print(f"\n🚀 开始训练，共 {EPOCHS} 轮...\n")
    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"训练 Loss:{train_loss:.4f} Acc:{train_acc:.3f} | "
              f"验证 Loss:{val_loss:.4f} Acc:{val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "best_model.pth"))

    # 7. 测试集评估
    print(f"\n📈 加载最优模型进行测试...")
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model.pth")))
    _, test_acc, preds, true_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\n✅ 测试集准确率：{test_acc:.4f}")
    print(f"\n📋 分类报告：")
    print(classification_report(
        true_labels, preds,
        target_names=["无火灾", "火灾"],
        zero_division=0))
    print("混淆矩阵：")
    cm = confusion_matrix(true_labels, preds)
    print(f"              预测无火灾  预测有火灾")
    print(f"实际无火灾  {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"实际有火灾  {cm[1][0]:>8}  {cm[1][1]:>8}")
    print(f"\n💾 最优模型已保存至：{MODEL_DIR}\\best_model.pth")


if __name__ == "__main__":
    main()
