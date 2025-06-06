# 语言检测器

这是一个基于传统机器学习方法的语言检测器，可以识别文本属于以下哪种语言：
- 德语 (de)
- 英语 (en)
- 西班牙语 (es)
- 法语 (fr)
- 意大利语 (it)
- 荷兰语 (nl)

## 项目结构

```
language_detector/
├── data/
│   ├── data.csv              # 原始数据集
│   └── processed/
│       ├── train.csv         # 训练集
│       └── test.csv          # 测试集
├── models/                   # 保存训练好的模型
├── split_data.py            # 数据集分割脚本
├── train_model.py           # 模型训练脚本
├── predict.py               # 预测脚本
└── requirements.txt         # 项目依赖
```

## 环境要求

- Python 3.6+
- 依赖包：
  - pandas >= 1.5.0
  - scikit-learn >= 1.0.0
  - numpy >= 1.21.0
  - joblib >= 1.1.0

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集

数据集包含9,066条文本样本，每种语言的分布如下：
- 西班牙语 (es): 1,562条
- 法语 (fr): 1,551条
- 意大利语 (it): 1,539条
- 英语 (en): 1,505条
- 德语 (de): 1,479条
- 荷兰语 (nl): 1,430条

数据集被随机分割为：
- 训练集：7,252条（80%）
- 测试集：1,814条（20%）

## 模型训练

我们尝试了两种特征提取方法和四种不同的分类器：

1. 特征提取方法：
   a. TF-IDF向量化：
      - 最大特征数：50,000
      - 使用单词和双词组合（n-gram范围：1-2）
      - 最小文档频率：2
      - 最大文档频率：95%
      - 统一转换为小写
      - 去除重音符号
   
   b. CountVectorizer：
      - 最大特征数：50,000
      - 使用单词和双词组合（n-gram范围：1-2）
      - 最小文档频率：2
      - 最大文档频率：95%
      - 统一转换为小写
      - 去除重音符号

2. 分类器比较：

| 特征提取方法 | 分类器 | 准确率 | 特点 |
|------------|--------|--------|------|
| TF-IDF | LinearSVC | 99.78% | 最佳模型，所有语言的精确率和召回率都在99%以上 |
| TF-IDF | MultinomialNB | 99.67% | 表现接近LinearSVC，计算速度快 |
| TF-IDF | LogisticRegression | 99.50% | 表现稳定，可解释性强 |
| TF-IDF | RandomForest | 96.03% | 表现相对较差，但具有特征重要性分析能力 |
| CountVectorizer | MultinomialNB | 99.67% | 与TF-IDF版本表现相当 |
| CountVectorizer | LinearSVC | 99.17% | 表现优秀，但略低于TF-IDF版本 |
| CountVectorizer | LogisticRegression | 98.84% | 表现稳定，但略低于TF-IDF版本 |
| CountVectorizer | RandomForest | 96.36% | 表现相对较差，但略好于TF-IDF版本 |

3. 最佳模型（TF-IDF + LinearSVC）的详细性能：

| 语言 | 精确率 | 召回率 | F1分数 |
|------|--------|--------|--------|
| 德语 (de) | 1.00 | 1.00 | 1.00 |
| 英语 (en) | 0.99 | 1.00 | 1.00 |
| 西班牙语 (es) | 1.00 | 1.00 | 1.00 |
| 法语 (fr) | 1.00 | 1.00 | 1.00 |
| 意大利语 (it) | 1.00 | 0.99 | 1.00 |
| 荷兰语 (nl) | 1.00 | 1.00 | 1.00 |

## 使用方法

1. 训练模型：
```bash
python train_model.py
```

2. 预测文本语言：
```bash
# 方式1：直接输入文本
python predict.py "这是一段要预测的文本"

# 方式2：通过管道输入
echo "这是一段要预测的文本" | python predict.py

# 方式3：交互式输入（按Ctrl+D结束输入）
python predict.py
```

## 实验结果分析

1. 模型表现：
   - 所有模型都达到了96%以上的准确率
   - TF-IDF + LinearSVC模型表现最好，准确率达到99.78%
   - CountVectorizer方法整体表现略低于TF-IDF方法
   - 所有语言类别的分类效果都很均衡

2. 特征提取方法比较：
   - TF-IDF方法整体表现更好，特别是在LinearSVC和LogisticRegression上
   - CountVectorizer方法在MultinomialNB上表现与TF-IDF相当
   - CountVectorizer方法在RandomForest上表现略好于TF-IDF
   - 两种方法都能很好地捕捉语言特征

3. 优势：
   - 模型轻量级，预测速度快
   - 不需要GPU资源
   - 训练和预测过程简单直观
   - 对六种语言的识别效果都很好
   - 提供了多种特征提取和分类器组合的选择

4. 可能的改进方向：
   - 尝试更多的特征提取方法（如字符级n-gram）
   - 使用深度学习模型（如BERT）进行对比
   - 增加更多语言的支持
   - 优化模型参数以提升性能
   - 添加模型解释性分析
   - 尝试特征选择方法
   - 探索其他分类器（如XGBoost、LightGBM等） 