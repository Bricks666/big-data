from flask import Flask, request
from joblib import load
from json import loads
from pathlib import Path
from os import getcwd
from pandas import DataFrame

model_path = Path(getcwd()) / 'src' / 'estates' / 'estates_regressor.pkl'


model = load(model_path)


app = Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'


@app.route('/predict/price', methods=['POST'])
def predict_price():
    body = loads(request.data)
    data = DataFrame(body)
    predicted = model.predict(data)

    return {'predicted': list(predicted)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
