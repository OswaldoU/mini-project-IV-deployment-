from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pandas as pd 
import pickle 


app = Flask(__name__)
api = Api(app)

model = pickle.load(open('model.pkl', 'rb'))

class Predict(Resource):
    def post(self): 
        json_data = request.get_json()
        df = pd.DataFrame(json_data)
        result = model.predict(df)
        return result.tolist()


api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug = True)