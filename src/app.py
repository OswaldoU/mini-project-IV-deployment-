from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pandas as pd 
import pickle 


app = Flask(__name__)
api = Api(app)

model = pickle.load(open('model.pkl', 'rb'))

# class Predict(Resource):
@app.route('/predict', methods = ["GET", "POST"])
def post(): 
    df = pd.read_csv("/Users/Oswal/Documents/GitHub/mini-project-IV-deployment-/notebooks/data.csv") 
    #json_data = request.get_json()
    #df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
    # getting predictions from our model.
    # it is much simpler because we used pipelines during development
    res = model.predict(df)
    print(res)
    # we cannot send numpy array as a result
    return jsonify(res.tolist())  

# api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug = True, host= '0.0.0.0')

@app.route('/', methods = ["GET", "POST"])
def index(): 
    return 'Hola'

    #  json_data = request.get_json()
    #    df = pd.DataFrame(json_data)
    #    result = model.predict(df)
    #    return result.tolist()