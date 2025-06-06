import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_data():
    """加载训练集和测试集"""
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    return train_df, test_df

def create_tfidf_pipeline(classifier):
    """创建包含TF-IDF向量化和分类器的管道"""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True
        )),
        ('classifier', classifier)
    ])

def create_count_pipeline(classifier):
    """创建包含CountVectorizer和分类器的管道"""
    return Pipeline([
        ('count', CountVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True
        )),
        ('classifier', classifier)
    ])

def train_and_evaluate_models():
    """训练和评估多个分类器模型"""
    # 加载数据
    train_df, test_df = load_data()
    X_train, y_train = train_df['text'], train_df['language']
    X_test, y_test = test_df['text'], test_df['language']

    # 定义要评估的分类器
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, n_jobs=-1),
        'LinearSVC': LinearSVC(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
        'MultinomialNB': MultinomialNB()
    }

    # 创建输出目录
    os.makedirs('models', exist_ok=True)
    
    # 训练和评估每个分类器（使用TF-IDF和CountVectorizer）
    results = {}
    vectorizers = {
        'tfidf': create_tfidf_pipeline,
        'count': create_count_pipeline
    }
    
    for vec_name, create_pipeline in vectorizers.items():
        print(f"\n使用 {vec_name.upper()} 向量化方法:")
        for name, classifier in classifiers.items():
            model_name = f"{vec_name}_{name}"
            print(f"\n训练 {model_name}...")
            
            # 创建并训练管道
            pipeline = create_pipeline(classifier)
            pipeline.fit(X_train, y_train)
            
            # 在测试集上评估
            y_pred = pipeline.predict(X_test)
            
            # 保存模型
            model_path = f'models/{model_name.lower()}_model.joblib'
            joblib.dump(pipeline, model_path)
            
            # 计算并存储结果
            results[model_name] = {
                'accuracy': pipeline.score(X_test, y_test),
                'report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"{model_name} 准确率: {results[model_name]['accuracy']:.4f}")
            print("\n分类报告:")
            print(results[model_name]['report'])

    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n最佳模型: {best_model[0]}, 准确率: {best_model[1]['accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    print("开始训练和评估模型...")
    results = train_and_evaluate_models() 