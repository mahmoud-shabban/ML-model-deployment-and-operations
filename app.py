import pickle
import numpy as np
from flask import Flask, render_template, make_response, url_for, jsonify
from flask_restx import Api, Resource, reqparse

# flask app
app = Flask(__name__)
app.config.from_object('config')
api = Api(app)
parser = reqparse.RequestParser()

# model load
model_path = "svc_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)


# prepocess form data to be fed into the model
def preprocess(data):
    d = np.array([data['pclass'], data['age'], data['sibsp'], data['parch']])
    if data['gender'] == 'male':
        t = [0, 1]
        d = np.append(d, t)
    elif data['gender'] == 'female':
        t = [1, 0]
        d = np.append(d, t)
    else:
        return f'The gender you passed is: {data["gender"]} it should take one of these values: [male , female]'
    return np.reshape(d, (1, len(d)))


# predict the processed data
def predict(data):
    p = model.predict_proba(data)  # [0]
    # if p == 1:
    #     return 'Will Survive!!'
    # else:
    #     return "Sorry About That!!"
    return p


# Routes
@api.route('/hello')
class Predict(Resource):
    def get(self):
        # headers = {'Content-Type': 'text/html'}
        response = make_response(render_template('index.html'))
        response.headers['Content-Type'] = 'text/html'
        return response

    # gender
    # parch
    # sibsp
    # pclass
    # age
    def post(self):
        parser.add_argument('gender', type=str, location='form')
        parser.add_argument('parch', type=int, location='form')
        parser.add_argument('sibsp', type=int, location='form')
        parser.add_argument('pclass', type=int, location='form')
        parser.add_argument('age', type=int, location='form')
        args = parser.parse_args()
        # print(type(args))
        # r = {i:args[i] for i in args}
        preprocessed_data = preprocess(args)
        predicttion = predict(preprocessed_data)

        response = make_response(render_template('result.html', prediction=predicttion))
        response.headers['Content-Type'] = 'text/html'

        return response
        # return jsonify({"type": str(args)})


if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])
