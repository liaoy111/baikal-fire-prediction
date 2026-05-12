# 🔥 贝加尔湖森林火灾风险预测与预警系统

基于深度学习（LSTM）的西伯利亚贝加尔湖地区火灾风险预测系统，整合多源遥感数据，实现历史火灾分析与未来7天风险预警。

---

## 📌 项目简介

本项目以俄罗斯西伯利亚贝加尔湖周边地区为研究区域，整合 NASA FIRMS 火点数据、Copernicus ERA5 气象数据及 MODIS NDVI 植被指数，构建 LSTM 时间序列预测模型，实现对火灾发生概率的预测，并结合实时天气预报 API 实现未来火灾风险自动预警。

---

## 🗂️ 项目结构

```
fire_project/
├── data/
│   ├── firms/                  # FIRMS火点数据
│   ├── weather/                # ERA5气象数据
│   ├── ndvi/                   # MODIS NDVI数据
│   └── processed_dataset.csv  # 预处理后的数据集
├── code/
│   ├── preprocess.py           # 数据预处理
│   ├── train_lstm_v2.py        # LSTM模型训练
│   ├── generate_map_data.py    # 生成可视化数据
│   └── predict_future.py       # 未来火灾风险预测
└── output/
    ├── best_model.pth          # 训练好的模型
    └── monthly_fire_risk.csv  # 月度风险数据
```

---

## 📦 数据来源

| 数据 | 来源 | 说明 |
|------|------|------|
| 火点数据 | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | MODIS Collection 6.1，每日火点记录 |
| 气象数据 | [Copernicus ERA5](https://cds.climate.copernicus.eu/) | 月均温度、风速、露点温度 |
| 植被指数 | [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/) | MOD13A2 NDVI，每16天 |
| 天气预报 | [Open-Meteo API](https://open-meteo.com/) | 未来7天实时预报，免费无需注册 |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
conda create -n fire_project python=3.10
conda activate fire_project
pip install torch torchvision rasterio numpy scikit-learn pandas xarray rioxarray netCDF4 scipy requests tqdm openpyxl
```

### 2. 准备数据

按照以下步骤下载数据：
- FIRMS：前往 [FIRMS下载页](https://firms.modaps.eosdis.nasa.gov/download/)，选择贝加尔湖范围（98°E, 49°N, 116°E, 58°N），下载2023年CSV格式数据
- ERA5：前往 [Copernicus CDS](https://cds.climate.copernicus.eu/)，下载ERA5-Land月均数据
- NDVI：前往 [AppEEARS](https://appeears.earthdatacloud.nasa.gov/)，下载MOD13A2产品

### 3. 修改路径

打开各代码文件，修改顶部配置区的路径为你的实际路径。

### 4. 运行流程

```bash
# 第一步：数据预处理
python code/preprocess.py

# 第二步：训练模型
python code/train_lstm_v2.py

# 第三步：生成可视化数据
python code/generate_map_data.py

# 第四步：预测未来火灾风险
python code/predict_future.py
```

---

## 📊 模型效果

| 指标 | 数值 |
|------|------|
| 测试集准确率 | 87% |
| 火灾召回率 | 97% |
| 火灾F1-score | 91% |

---

## 🔮 预警输出示例

```
🚀 火灾风险预警系统
=======================================================
🔮 贝加尔湖未来7天火灾风险预测
=======================================================
日期         温度      风速    火灾概率  风险等级
2026-05-14  -1.1°C  17.9km/h   0.76   🔴 高风险
2026-05-15  -1.6°C  41.0km/h   0.81   🔴 高风险
2026-05-16  -1.4°C  14.2km/h   0.83   🔴 高风险
=======================================================
⚡ 提示：未来一周存在一定火灾风险，请注意关注。
```

---

## 🗺️ 可视化

月度火灾风险地图使用 ArcGIS Pro 制作，支持时间序列动态展示：

- 🟢 低风险（1-2月，11-12月）
- 🟠 中风险（3月）
- 🔴 高风险（4-10月）

---

## 🛠️ 技术栈

- **深度学习**：PyTorch、LSTM
- **数据处理**：pandas、numpy、rasterio、xarray
- **GIS可视化**：ArcGIS Pro
- **数据来源**：NASA FIRMS、Copernicus ERA5、MODIS NDVI
- **天气预报**：Open-Meteo API

---

## 👤 作者

地理信息科学专业本科生
如有问题欢迎提 Issue 或联系我 😊
