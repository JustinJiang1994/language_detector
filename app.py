from flask import Flask, request, jsonify, render_template
import joblib
import os
from predict import predict_language

app = Flask(__name__)

# 确保模板目录存在
os.makedirs('templates', exist_ok=True)

@app.route('/')
def home():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_language():
    """语言检测API接口"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': '请提供要检测的文本',
                'status': 'error'
            }), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({
                'error': '文本不能为空',
                'status': 'error'
            }), 400
        
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
        
        return jsonify({
            'text': text,
            'language': language,
            'language_name': language_names.get(language, language),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """获取支持的语言列表"""
    languages = {
        'de': '德语',
        'en': '英语',
        'es': '西班牙语',
        'fr': '法语',
        'it': '意大利语',
        'nl': '荷兰语'
    }
    return jsonify({
        'languages': languages,
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 