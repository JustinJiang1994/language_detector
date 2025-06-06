import joblib
import sys

def load_model():
    """加载最佳模型"""
    return joblib.load('models/linearsvc_model.joblib')

def predict_language(text):
    """预测文本的语言"""
    model = load_model()
    prediction = model.predict([text])[0]
    return prediction

def main():
    if len(sys.argv) > 1:
        # 从命令行参数获取文本
        text = ' '.join(sys.argv[1:])
    else:
        # 从标准输入获取文本
        print("请输入要预测的文本（输入完成后按Ctrl+D结束）：")
        text = sys.stdin.read().strip()
    
    if not text:
        print("错误：未提供输入文本")
        sys.exit(1)
    
    # 预测语言
    language = predict_language(text)
    
    # 语言代码到名称的映射
    language_names = {
        'de': '德语',
        'en': '英语',
        'es': '西班牙语',
        'fr': '法语',
        'it': '意大利语',
        'nl': '荷兰语'
    }
    
    print(f"\n预测结果：{language_names.get(language, language)}")

if __name__ == "__main__":
    main() 