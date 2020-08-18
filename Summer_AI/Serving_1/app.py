# 데이터 전처리 -> 학습 -> save 해서 ~.h5로 저장
# 학습된 모델을 다른 파일로 옮겨서 load_model 하면 거기서도 사용 가능
# 결과를 웹에서 보려고 할 때 사용

# python warning off
import warnings
warnings.filterwarnings('ignore')

# tensorflow warning off
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn import datasets
from keras.models import load_model

app = Flask(__name__)

global model
global graph
global target_names

model = load_model("iris_model.h5")
graph = tf.get_default_graph()
target_names = datasets.load_iris()['target_names']

# 예) https://www.naver.com/
# http://localhost:5000/
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hi")
def hi():
    return "hi world"

@app.route("/checkform")
def checkform():
    return render_template("checkform.html")

@app.route("/check", methods=['POST'])
def check():
    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']
    features = [sepal_length, sepal_width, petal_length, petal_width]
    print("features:", features)
    features = np.reshape(features, (1, 4))
    with graph.as_default():
        y_pred = model.predict_classes(features)
        
    print({'species': target_names[y_pred[0]]})


    return jsonify( {'species': target_names[y_pred[0]]} )




if __name__ == "__main__":
    app.run(debug=True)

# 개발모드 on --> debug=True
# 런칭 시 --> debug=False