# 传媒广告时间序列分析 - 修复列识别错误
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("传媒广告时间序列分析与预测系统")
print("=" * 60)

# 1. 读取数据
print("\n步骤1: 读取数据")
try:
    df = pd.read_excel('Marketing Campaign Analysis.xlsx')
    print(f"✓ 数据读取成功，形状: {df.shape}")
except Exception as e:
    print(f"读取文件失败: {e}")
    exit()

# 2. 手动指定正确的列名（基于你的数据列名）
print("\n步骤2: 列名指定")
# 根据你的数据列名，手动指定正确的列
date_col = 'time'  # 日期列
ctr_col = 'CTR (%)'  # CTR列
impression_col = 'impressions'  # 展示量列
click_col = 'clicks'  # 点击量列
cpc_col = 'CPC (USD)'  # CPC列
channel_col = 'channel_name'  # 渠道列

print(f"使用的列名:")
print(f"- 日期列: {date_col}")
print(f"- CTR列: {ctr_col}")
print(f"- 展示量列: {impression_col}")
print(f"- 点击量列: {click_col}")
print(f"- CPC列: {cpc_col}")
print(f"- 渠道列: {channel_col}")

# 3. 数据清洗和预处理
print("\n步骤3: 数据清洗")

# 选择需要的列
selected_cols = [date_col, channel_col, impression_col, click_col, ctr_col, cpc_col]
selected_cols = [col for col in selected_cols if col in df.columns]

df_clean = df[selected_cols].copy()

# 检查日期列的数据类型
print(f"日期列 '{date_col}' 的数据类型: {df_clean[date_col].dtype}")
print(f"日期列前5个值: {df_clean[date_col].head().tolist()}")

# 转换日期列
try:
    # 尝试直接转换日期
    df_clean['date'] = pd.to_datetime(df_clean[date_col])
    print("✓ 日期转换成功")
except Exception as e:
    print(f"日期转换失败: {e}")
    # 如果转换失败，尝试其他方法
    try:
        # 可能是Excel序列号，尝试用origin转换
        df_clean['date'] = pd.to_datetime(df_clean[date_col], unit='D', origin='1899-12-30')
        print("✓ 使用Excel序列号转换日期成功")
    except:
        print("❌ 日期转换失败，请检查数据格式")
        exit()

# 删除日期为空的行
df_clean = df_clean.dropna(subset=['date'])
print(f"清洗后数据量: {len(df_clean)}")

# 4. 数据聚合（按天）
print("\n步骤4: 数据聚合")
daily_data = df_clean.groupby('date').agg({
    impression_col: 'sum',
    click_col: 'sum',
    ctr_col: 'mean',
    cpc_col: 'mean'
}).reset_index()

# 计算每日CTR（以防原始CTR列有问题）
daily_data['daily_ctr'] = (daily_data[click_col] / daily_data[impression_col] * 100).round(3)
daily_data = daily_data.sort_values('date')

print(f"聚合后数据天数: {len(daily_data)}")
print("前5天数据:")
print(daily_data[['date', impression_col, click_col, 'daily_ctr']].head())

# 5. 基本统计分析
print("\n步骤5: 统计分析")
print(f"分析时间段: {daily_data['date'].min()} 到 {daily_data['date'].max()}")
print(f"总天数: {len(daily_data)}")
print(f"平均CTR: {daily_data['daily_ctr'].mean():.3f}%")
print(f"CTR范围: {daily_data['daily_ctr'].min():.3f}% - {daily_data['daily_ctr'].max():.3f}%")
print(f"总展示量: {daily_data[impression_col].sum():,}")
print(f"总点击量: {daily_data[click_col].sum():,}")

# 6. 时间序列可视化
print("\n步骤6: 数据可视化")
plt.figure(figsize=(15, 10))

# 6.1 主要指标趋势
plt.subplot(2, 2, 1)
plt.plot(daily_data['date'], daily_data['daily_ctr'], marker='o', linewidth=1, markersize=3)
plt.title('每日CTR趋势')
plt.ylabel('CTR (%)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(daily_data['date'], daily_data[impression_col], color='orange', linewidth=1)
plt.title('每日展示量')
plt.ylabel('展示量')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(daily_data['date'], daily_data[click_col], color='green', linewidth=1)
plt.title('每日点击量')
plt.ylabel('点击量')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(daily_data['date'], daily_data[cpc_col], color='red', linewidth=1)
plt.title('每日CPC')
plt.ylabel('CPC (USD)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('广告趋势分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 周分析
print("\n步骤7: 周度分析")
daily_data['weekday'] = daily_data['date'].dt.day_name()
weekday_ctr = daily_data.groupby('weekday')['daily_ctr'].mean()

# 按周顺序排序
week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_ctr = weekday_ctr.reindex(week_order)

plt.figure(figsize=(10, 6))
plt.bar(weekday_ctr.index, weekday_ctr.values, color='lightblue')
plt.title('各工作日平均CTR对比')
plt.ylabel('平均CTR (%)')
plt.xlabel('星期')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('周度分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 简单预测模型
print("\n步骤8: 建立预测模型")

# 准备时间序列数据
ts_data = daily_data.set_index('date')['daily_ctr']

# 划分训练集和测试集
train_size = int(len(ts_data) * 0.8)
train_data = ts_data.iloc[:train_size]
test_data = ts_data.iloc[train_size:]

print(f"训练集: {len(train_data)}天, 测试集: {len(test_data)}天")


# 简单移动平均预测
def moving_average_forecast(data, window=7):
    """移动平均预测"""
    return data.rolling(window=window).mean().iloc[-1]


# 对测试集进行预测
test_predictions = []
for i in range(len(test_data)):
    if i < 7:
        # 前7天使用训练集最后7天的平均值
        pred = train_data.tail(7).mean()
    else:
        # 使用前7天的实际值计算移动平均
        available_data = pd.concat([train_data, test_data.iloc[:i]])
        pred = moving_average_forecast(available_data, 7)
    test_predictions.append(pred)

test_predictions = pd.Series(test_predictions, index=test_data.index)

# 计算预测误差
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test_data, test_predictions)
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))

print(f"模型预测效果:")
print(f"平均绝对误差(MAE): {mae:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")

# 9. 未来预测
print("\n步骤9: 未来预测")

# 使用全部数据预测未来7天
future_days = 7
future_predictions = []

# 使用最后7天的移动平均作为未来预测基础
current_data = ts_data.copy()
for i in range(future_days):
    next_pred = moving_average_forecast(current_data, 7)
    future_predictions.append(next_pred)
    # 将预测值添加到数据中用于后续预测（模拟未来数据）
    next_date = current_data.index[-1] + timedelta(days=1)
    current_data[next_date] = next_pred

future_dates = [ts_data.index[-1] + timedelta(days=i + 1) for i in range(future_days)]

print("\n未来7天CTR预测:")
future_df = pd.DataFrame({
    '日期': future_dates,
    '预测CTR(%)': [f"{x:.3f}%" for x in future_predictions]
})
print(future_df)

# 10. 可视化预测结果
print("\n步骤10: 可视化结果")
plt.figure(figsize=(14, 8))

# 历史数据
plt.plot(ts_data.index, ts_data.values, label='历史CTR', color='blue', linewidth=2)

# 测试集预测
plt.plot(test_predictions.index, test_predictions.values,
         label='测试集预测', color='orange', linestyle='--', linewidth=2)

# 未来预测
plt.plot(future_dates, future_predictions,
         label='未来预测', color='red', marker='o', linewidth=2)

plt.title('CTR时间序列分析与预测', fontsize=14)
plt.xlabel('日期')
plt.ylabel('CTR (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('CTR预测结果.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 渠道分析
print("\n步骤11: 渠道效果分析")

if channel_col in df_clean.columns:
    channel_analysis = df_clean.groupby(channel_col).agg({
        impression_col: 'sum',
        click_col: 'sum',
        ctr_col: 'mean',
        cpc_col: 'mean'
    }).round(3)

    channel_analysis['计算CTR'] = (channel_analysis[click_col] / channel_analysis[impression_col] * 100).round(3)
    channel_analysis = channel_analysis.sort_values('计算CTR', ascending=False)

    print("\n各渠道表现汇总:")
    print(channel_analysis[['计算CTR', cpc_col, impression_col, click_col]])

    # 渠道可视化
    plt.figure(figsize=(12, 6))
    plt.bar(channel_analysis.index.astype(str), channel_analysis['计算CTR'])
    plt.title('各渠道CTR对比')
    plt.ylabel('CTR (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('渠道分析.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("未找到渠道列，跳过渠道分析")

# 12. 生成分析报告
print("\n" + "=" * 60)
print("分析报告总结")
print("=" * 60)

# 计算关键指标
ctr_trend = "上升" if ts_data.iloc[-1] > ts_data.iloc[0] else "下降"
ctr_change_pct = ((ts_data.iloc[-1] - ts_data.iloc[0]) / ts_data.iloc[0] * 100)

best_weekday = weekday_ctr.idxmax()
worst_weekday = weekday_ctr.idxmin()

print(f"""
📊 趋势分析:
   • CTR整体{ctr_trend}趋势: {abs(ctr_change_pct):.1f}%
   • 最佳表现日期: {best_weekday} (平均CTR: {weekday_ctr[best_weekday]:.3f}%)
   • 最差表现日期: {worst_weekday} (平均CTR: {weekday_ctr[worst_weekday]:.3f}%)

📈 预测结果:
   • 模型预测精度(MAE): {mae:.4f}
   • 未来7天平均CTR预测: {np.mean(future_predictions):.3f}%
   • 预测趋势: {'乐观' if np.mean(future_predictions) > ts_data.mean() else '保守'}

💡 优化建议:
   • 在{best_weekday}加大广告投放力度
   • 关注CTR波动，建立异常监测机制
   • 定期更新预测模型，适应市场变化
""")

# 13. 保存结果
print("\n步骤12: 保存结果")

# 保存预测结果
results_df = pd.DataFrame({
    '日期': list(ts_data.index) + future_dates,
    'CTR': list(ts_data.values) + future_predictions,
    '类型': ['历史'] * len(ts_data) + ['预测'] * len(future_dates)
})

results_df.to_excel('CTR时间序列分析结果.xlsx', index=False)
print("✓ 分析结果已保存至: CTR时间序列分析结果.xlsx")

# 保存汇总统计
stats_df = pd.DataFrame({
    '指标': ['总天数', '平均CTR', '最高CTR', '最低CTR', '总展示量', '总点击量', '预测MAE'],
    '数值': [
        len(daily_data),
        f"{daily_data['daily_ctr'].mean():.3f}%",
        f"{daily_data['daily_ctr'].max():.3f}%",
        f"{daily_data['daily_ctr'].min():.3f}%",
        f"{daily_data[impression_col].sum():,}",
        f"{daily_data[click_col].sum():,}",
        f"{mae:.4f}"
    ]
})

stats_df.to_csv('分析统计汇总.csv', index=False, encoding='utf-8-sig')
print("✓ 统计汇总已保存至: 分析统计汇总.csv")
print("✓ 图表已保存为PNG文件")

print("\n" + "=" * 60)
print("✅ 分析完成！")
print("=" * 60)