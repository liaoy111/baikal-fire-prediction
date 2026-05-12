"""
第4步：预测未来火灾风险
自动从 Open-Meteo 获取贝加尔湖未来7天天气预报
用训练好的LSTM模型预测火灾风险
"""

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ⚙️ 配置区
# ============================================================

DATA_FILE  = r"D:\PycharmProject\fire_project\data\processed_dataset.csv"
MODEL_PATH = r"D:\PycharmProject\fire_project\output\best_model.pth"

# 贝加尔湖坐标
LAT = 53.5
LON = 108.0

SEQUENCE_LEN = 7
FEATURE_COLS = ['temperature', 'dewpoint', 'wind_speed', 'ndvi']

# ============================================================
# LSTM模型定义（和训练时一样）
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
# 第一部分：从 Open-Meteo 获取天气预报
# ============================================================

def get_weather_forecast():
    print("🌤️  正在获取贝加尔湖天气预报...")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":        LAT,
        "longitude":       LON,
        "daily": [
            "temperature_2m_max",      # 最高温度
            "temperature_2m_min",      # 最低温度
            "dewpoint_2m_max",         # 露点温度
            "wind_speed_10m_max",      # 最大风速
        ],
        "timezone":        "Asia/Irkutsk",
        "forecast_days":   7
    }

    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        print(f"❌ 获取天气数据失败，状态码：{response.status_code}")
        return None

    data = response.json()
    daily = data['daily']

    df = pd.DataFrame({
        'date':        pd.to_datetime(daily['time']),
        'temperature': [(a + b) / 2 for a, b in zip(
                        daily['temperature_2m_max'],
                        daily['temperature_2m_min'])],  # 取最高最低平均
        'dewpoint':    daily['dewpoint_2m_max'],
        'wind_speed':  daily['wind_speed_10m_max'],
    })

    print(f"✅ 获取成功！未来 {len(df)} 天的天气预报：\n")
    print(f"{'日期':<12} {'温度(°C)':>8} {'露点(°C)':>8} {'风速(km/h)':>10}")
    print("-" * 45)
    for _, row in df.iterrows():
        print(f"{str(row['date'].date()):<12} "
              f"{row['temperature']:>8.1f} "
              f"{row['dewpoint']:>8.1f} "
              f"{row['wind_speed']:>10.1f}")

    return df


# ============================================================
# 第二部分：准备预测数据
# ============================================================

def prepare_prediction_data(forecast_df):
    """
    用历史数据的最后几天 + 天气预报数据组合成预测序列
    NDVI用历史数据最近一期的值
    """
    # 读取历史数据
    hist_df = pd.read_csv(DATA_FILE)
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date').reset_index(drop=True)

    # 用历史数据最近的NDVI值
    latest_ndvi = hist_df['ndvi'].iloc[-1]
    forecast_df['ndvi'] = latest_ndvi

    # 用历史数据训练归一化器
    scaler = StandardScaler()
    scaler.fit(hist_df[FEATURE_COLS])

    # 归一化预报数据
    forecast_scaled = scaler.transform(forecast_df[FEATURE_COLS])

    # 取历史数据最后(SEQUENCE_LEN-1)天作为前置序列
    hist_scaled = scaler.transform(hist_df[FEATURE_COLS])
    prefix = hist_scaled[-(SEQUENCE_LEN-1):]  # 取最后6天历史数据

    # 对每一天预测：用前6天历史+当天预报
    sequences = []
    for i in range(len(forecast_df)):
        if i == 0:
            seq = np.vstack([prefix, forecast_scaled[0:1]])
        else:
            seq = np.vstack([
                prefix[i:],
                forecast_scaled[:i+1]
            ])
            # 如果长度不够就用预报数据补
            while len(seq) < SEQUENCE_LEN:
                seq = np.vstack([forecast_scaled[0:1], seq])
        sequences.append(seq[:SEQUENCE_LEN])

    return np.array(sequences), forecast_df


# ============================================================
# 第三部分：预测并输出结果
# ============================================================

def predict_fire_risk(sequences, forecast_df):
    device = torch.device("cpu")
    model = FireLSTM(input_size=len(FEATURE_COLS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        X = torch.tensor(sequences, dtype=torch.float32)
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
        preds = (probs > 0.5).astype(int)

    # 风险等级
    def risk_level(p):
        if p >= 0.7:   return '🔴 高风险'
        elif p >= 0.4: return '🟠 中风险'
        else:          return '🟢 低风险'

    print("\n" + "="*55)
    print("🔮 贝加尔湖未来7天火灾风险预测")
    print("="*55)
    print(f"{'日期':<12} {'温度':>6} {'风速':>8} {'火灾概率':>8} {'风险等级'}")
    print("-"*55)

    for i, (_, row) in enumerate(forecast_df.iterrows()):
        print(f"{str(row['date'].date()):<12} "
              f"{row['temperature']:>5.1f}°C "
              f"{row['wind_speed']:>6.1f}km/h "
              f"{probs[i]:>8.2f}   "
              f"{risk_level(probs[i])}")

    print("="*55)

    # 整体风险评估
    avg_prob = probs.mean()
    high_risk_days = sum(preds)
    print(f"\n📊 整体评估：")
    print(f"   未来7天平均火灾概率：{avg_prob:.2f}")
    print(f"   预测高风险天数：{high_risk_days} 天")

    if avg_prob >= 0.7:
        print(f"\n⚠️  警告：未来一周火灾风险极高，建议加强监测！")
    elif avg_prob >= 0.4:
        print(f"\n⚡ 提示：未来一周存在一定火灾风险，请注意关注。")
    else:
        print(f"\n✅ 未来一周火灾风险较低。")


# ============================================================
# 主流程
# ============================================================

def main():
    print("="*55)
    print("🚀 火灾风险预警系统")
    print("="*55)

    # 1. 获取天气预报
    forecast_df = get_weather_forecast()
    if forecast_df is None:
        return

    # 2. 准备预测数据
    sequences, forecast_df = prepare_prediction_data(forecast_df)

    # 3. 预测并输出
    predict_fire_risk(sequences, forecast_df)


if __name__ == "__main__":
    main()
