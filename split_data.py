import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 读取数据
df = pd.read_csv('data/data.csv', header=None, names=['text', 'language'])

# 确保输出目录存在
os.makedirs('data/processed', exist_ok=True)

# 将数据分割成训练集（80%）和测试集（20%）
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['language'])

# 保存分割后的数据集
train_df.to_csv('data/processed/train.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

# 打印数据集的基本信息
print(f"总样本数: {len(df)}")
print(f"训练集样本数: {len(train_df)}")
print(f"测试集样本数: {len(test_df)}")
print("\n语言分布:")
print(df['language'].value_counts()) 