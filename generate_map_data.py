"""
第3步：生成ArcGIS Pro可视化数据
输出：按月统计的火灾风险CSV，可以在ArcGIS Pro里制作时间序列地图
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

# ============================================================
# ⚙️ 配置区
# ============================================================

DATA_FILE   = r"D:\PycharmProject\fire_project\data\processed_dataset.csv"
MODEL_PATH  = r"D:\PycharmProject\fire_project\output\best_model.pth"
OUTPUT_DIR  = r"D:\PycharmProject\fire_project\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQUENCE_LEN = 7
FEATURE_COLS = ['temperature', 'dewpoint', 'wind_speed', 'ndvi']

# 贝加尔湖中心坐标（用于地图显示）
BAIKAL_LAT = 53.5
BAIKAL_LON = 108.0

# ============================================================
# 模型定义（和训练时一样）
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
# 主流程
# ============================================================

def main():
    print("="*50)
    print("🗺️  生成ArcGIS Pro可视化数据")
    print("="*50)

    # 1. 读取数据
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. 归一化（和训练时一样）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    # 3. 创建序列
    X, dates = [], []
    for i in range(SEQUENCE_LEN, len(df)):
        X.append(df[FEATURE_COLS].iloc[i-SEQUENCE_LEN:i].values)
        dates.append(df['date'].iloc[i])
    X = np.array(X)
    dates = pd.Series(dates)

    # 4. 加载模型并预测
    device = torch.device("cpu")
    model = FireLSTM(input_size=len(FEATURE_COLS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()  # 火灾概率
        preds = (probs > 0.5).astype(int)

    # 5. 整合预测结果
    results = pd.DataFrame({
        'date':           dates.values,
        'fire_prob':      probs,           # 火灾概率（0-1）
        'fire_predicted': preds,           # 预测标签（0/1）
        'fire_actual':    df['fire_label'].iloc[SEQUENCE_LEN:].values,
        'temperature':    df['temperature'].iloc[SEQUENCE_LEN:].values,
        'ndvi':           df['ndvi'].iloc[SEQUENCE_LEN:].values,
    })
    results['date'] = pd.to_datetime(results['date'])
    results['month'] = results['date'].dt.month
    results['month_name'] = results['date'].dt.strftime('%Y-%m')

    # 6. 按月统计
    monthly = results.groupby('month_name').agg(
        fire_prob_mean    = ('fire_prob',      'mean'),   # 月平均火灾概率
        fire_prob_max     = ('fire_prob',      'max'),    # 月最高火灾概率
        fire_days_pred    = ('fire_predicted', 'sum'),    # 预测火灾天数
        fire_days_actual  = ('fire_actual',    'sum'),    # 实际火灾天数
        temperature_mean  = ('temperature',    'mean'),   # 月平均温度
        ndvi_mean         = ('ndvi',           'mean'),   # 月平均NDVI
    ).reset_index()

    # 7. 添加风险等级
    def risk_level(prob):
        if prob >= 0.7:   return '高风险'
        elif prob >= 0.4: return '中风险'
        else:             return '低风险'

    monthly['risk_level'] = monthly['fire_prob_mean'].apply(risk_level)

    # 8. 添加贝加尔湖坐标（ArcGIS Pro显示用）
    monthly['latitude']  = BAIKAL_LAT
    monthly['longitude'] = BAIKAL_LON

    # 9. 保存结果
    output_path = os.path.join(OUTPUT_DIR, "monthly_fire_risk.csv")
    monthly.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 10. 打印结果
    print("\n📊 按月火灾风险统计：\n")
    print(f"{'月份':<10} {'火灾概率':>8} {'风险等级':>8} {'预测火灾天':>10} {'实际火灾天':>10}")
    print("-" * 55)
    for _, row in monthly.iterrows():
        print(f"{row['month_name']:<10} "
              f"{row['fire_prob_mean']:>8.2f} "
              f"{row['risk_level']:>8} "
              f"{int(row['fire_days_pred']):>10} "
              f"{int(row['fire_days_actual']):>10}")

    print(f"\n💾 数据已保存至：{output_path}")
    print("\n✅ 完成！现在可以在ArcGIS Pro里导入这个CSV文件")

if __name__ == "__main__":
    main()
