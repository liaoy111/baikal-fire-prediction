"""
第1步：数据预处理
读取 FIRMS、ERA5气象、NDVI 三类数据
清洗并整合成一个可以用于机器学习的数据集
"""

import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import glob
import os
import re
from datetime import datetime, timedelta

# ============================================================
# ⚙️ 路径配置 — 确认路径是否正确！
# ============================================================

BASE_DIR      = r"D:\PycharmProject\fire_project\data"
FIRMS_FILE = os.path.join(BASE_DIR, r"firms\fire_archive_M-C61_748065.csv")
WEATHER_FILE  = os.path.join(BASE_DIR, r"weather\ERA5_weather_baikal_2023.nc")
NDVI_DIR      = os.path.join(BASE_DIR, r"ndvi")
OUTPUT_FILE   = r"D:\PycharmProject\fire_project\data\processed_dataset.csv"

# ============================================================
# 第一部分：读取并清洗 FIRMS 火点数据
# ============================================================

def process_firms():
    print("\n" + "="*50)
    print("🔥 第一部分：处理 FIRMS 火点数据")
    print("="*50)

    df = pd.read_csv(FIRMS_FILE)
    print(f"原始数据：{len(df)} 条记录")

    # 1. 统一日期格式
    df['acq_date'] = pd.to_datetime(df['acq_date'])

    # 2. 去掉低置信度数据（置信度 < 50 的不可靠）
    df = df[df['confidence'] >= 50]
    print(f"去除低置信度后：{len(df)} 条记录")

    # 3. 去掉 FRP 为 0 的异常数据
    df = df[df['frp'] > 0]
    print(f"去除FRP=0后：{len(df)} 条记录")

    # 4. 只保留需要的列
    df = df[['acq_date', 'latitude', 'longitude', 'brightness', 'frp', 'daynight']]

    # 5. 按日期统计每天的火点数量和平均强度
    daily_fire = df.groupby('acq_date').agg(
        fire_count   = ('frp', 'count'),    # 每天火点数量
        fire_frp_mean= ('frp', 'mean'),     # 每天平均火灾强度
        fire_frp_max = ('frp', 'max'),      # 每天最大火灾强度
    ).reset_index()

    # 6. 生成完整的日期序列（没有火灾的日期也要有记录）
    all_dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    daily_fire = daily_fire.set_index('acq_date').reindex(all_dates).reset_index()
    daily_fire.columns = ['date', 'fire_count', 'fire_frp_mean', 'fire_frp_max']

    # 7. 没有火灾的日期填0
    daily_fire = daily_fire.fillna(0)

    # 8. 创建标签：当天火点数量 > 0 就是火灾日
    daily_fire['fire_label'] = (daily_fire['fire_count'] > 0).astype(int)

    print(f"\n✅ FIRMS处理完成！")
    print(f"   火灾天数：{daily_fire['fire_label'].sum()} 天")
    print(f"   无火灾天数：{(daily_fire['fire_label']==0).sum()} 天")

    return daily_fire


# ============================================================
# 第二部分：读取并处理 ERA5 气象数据
# ============================================================

def process_weather():
    print("\n" + "="*50)
    print("🌡️  第二部分：处理 ERA5 气象数据")
    print("="*50)

    ds = xr.open_dataset(WEATHER_FILE)
    print(f"气象数据变量：{list(ds.data_vars)}")
    print(f"所有维度：{dict(ds.dims)}")
    print(f"时间范围：{ds.valid_time.values[0]} ~ {ds.valid_time.values[-1]}")

    # 对整个贝加尔湖区域取平均值，得到每月一个代表值
    weather_mean = ds.mean(dim=['latitude', 'longitude'])
    df_weather = weather_mean.to_dataframe().reset_index()
    df_weather = df_weather.rename(columns={'valid_time': 'time'})

    # 重命名列（ERA5变量名比较长）
    rename_map = {}
    for col in df_weather.columns:
        if 't2m'  in col: rename_map[col] = 'temperature'    # 温度（单位：K）
        if 'd2m'  in col: rename_map[col] = 'dewpoint'       # 露点温度
        if 'u10'  in col: rename_map[col] = 'wind_u'         # 东西风速
        if 'v10'  in col: rename_map[col] = 'wind_v'         # 南北风速
    df_weather = df_weather.rename(columns=rename_map)

    # 温度从开尔文转换为摄氏度
    if 'temperature' in df_weather.columns:
        df_weather['temperature'] = df_weather['temperature'] - 273.15
    if 'dewpoint' in df_weather.columns:
        df_weather['dewpoint'] = df_weather['dewpoint'] - 273.15

    # 计算风速（合并东西和南北分量）
    if 'wind_u' in df_weather.columns and 'wind_v' in df_weather.columns:
        df_weather['wind_speed'] = np.sqrt(
            df_weather['wind_u']**2 + df_weather['wind_v']**2
        )

    # 重命名时间列
    df_weather = df_weather.rename(columns={'time': 'date'})
    df_weather['date'] = pd.to_datetime(df_weather['date'])

    print(f"\n✅ 气象数据处理完成！")
    print(f"   共 {len(df_weather)} 个月的数据")
    print(df_weather[['date', 'temperature', 'wind_speed']].head())

    return df_weather


# ============================================================
# 第三部分：读取并处理 NDVI 数据
# ============================================================

def process_ndvi():
    print("\n" + "="*50)
    print("🌿 第三部分：处理 NDVI 植被数据")
    print("="*50)

    tif_files = sorted(glob.glob(os.path.join(NDVI_DIR, "*NDVI*.tif")))
    print(f"找到 {len(tif_files)} 个NDVI文件")

    records = []
    for f in tif_files:
        # 从文件名提取日期
        match = re.search(r'(\d{8})T', os.path.basename(f))
        if not match:
            continue
        date_str = match.group(1)
        date = datetime.strptime(date_str, '%Y%m%d')

        # 读取TIF并计算区域平均NDVI
        try:
            da = rioxarray.open_rasterio(f, masked=True)
            # MODIS NDVI 原始值需要乘以0.0001转换为真实值（范围-1到1）
            ndvi_mean = float(da.mean().values) * 0.0001
            records.append({'date': date, 'ndvi': ndvi_mean})
        except Exception as e:
            print(f"  ⚠️  读取失败：{os.path.basename(f)}，原因：{e}")

    df_ndvi = pd.DataFrame(records)
    df_ndvi['date'] = pd.to_datetime(df_ndvi['date'])
    print(f"\n✅ NDVI处理完成！")
    print(f"   共 {len(df_ndvi)} 个时间点的数据")
    print(f"   NDVI范围：{df_ndvi['ndvi'].min():.3f} ~ {df_ndvi['ndvi'].max():.3f}")

    return df_ndvi


# ============================================================
# 第四部分：整合所有数据
# ============================================================

def merge_all(firms_df, weather_df, ndvi_df):
    print("\n" + "="*50)
    print("🔗 第四部分：整合所有数据")
    print("="*50)

    df = firms_df.copy()

    # 整合气象数据（月度数据，按月匹配）
    df['month'] = df['date'].dt.to_period('M')
    weather_df['month'] = weather_df['date'].dt.to_period('M')
    weather_cols = [c for c in weather_df.columns
                    if c in ['temperature', 'dewpoint', 'wind_speed', 'wind_u', 'wind_v', 'month']]
    df = df.merge(weather_df[weather_cols], on='month', how='left')

    # 整合NDVI数据（16天数据，用最近一期匹配）
    def find_nearest_ndvi(date, ndvi_df):
        diff = (ndvi_df['date'] - date).abs()
        return ndvi_df.loc[diff.idxmin(), 'ndvi']

    df['ndvi'] = df['date'].apply(lambda d: find_nearest_ndvi(d, ndvi_df))

    # 删除辅助列
    df = df.drop(columns=['month'])

    # 删除有缺失值的行
    before = len(df)
    df = df.dropna()
    print(f"删除缺失值：{before - len(df)} 行")

    print(f"\n✅ 整合完成！最终数据集：{len(df)} 行，{len(df.columns)} 列")
    print(f"\n列名：{list(df.columns)}")
    print(f"\n前5行预览：")
    print(df.head())

    return df


# ============================================================
# 主流程
# ============================================================

def main():
    print("="*50)
    print("🚀 火灾预测数据预处理开始")
    print("="*50)

    firms_df   = process_firms()
    weather_df = process_weather()
    ndvi_df    = process_ndvi()
    final_df   = merge_all(firms_df, weather_df, ndvi_df)

    # 保存最终数据集
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 数据集已保存至：{OUTPUT_FILE}")
    print("\n🎉 预处理完成！")


if __name__ == "__main__":
    main()
