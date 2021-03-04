from datetime import datetime
import json

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
from prometheus_client import Counter
from prometheus_flask_exporter import PrometheusMetrics
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from model import Model
from persistence import TrainModel
from parameters import model_path, pipeline_name, params

total_predictions = Counter('predictions',
                            'Total number of predictions',
                            ['model_name', 'date'])
total_fraud_predictions = Counter('fraud_predictions',
                                  'Total number of fraud_predictions')

app = Flask(__name__)
api = Api(app)
metrics = PrometheusMetrics(app)


class Prediction(Resource):

    @staticmethod
    def post():
        try:
            email = request.json.get('email', '')
            model_name = request.json.get('model_name', pipeline_name)
            query_df = pd.DataFrame([{'email': email}])
            model = Model.load_pipeline(model_path, model_name)
            prediction = model.predict(query_df).tolist()
            labels = {'model_name': model_name,
                      'date': datetime.now()}
            if prediction[0]:
                total_fraud_predictions.inc()
            total_predictions.labels(**labels).inc()
            return jsonify({'prediction for {}'.format(email): prediction,
                            'model_name': model_name
                            })
        except Exception as e:
            return jsonify({'error': str(e)})


class Training(Resource):

    @staticmethod
    def post():
        try:
            model_params = request.json.get('model_params')
            model_name = request.json.get('model_name')
            new_model = Model.get_model(model_params, model_name)
            accuracy = new_model[1]
            date = datetime.now()
            new_model = TrainModel.add(name=model_name,
                                       model_params=json.dumps(model_params),
                                       accuracy=accuracy,
                                       serving=False,
                                       train_date=date)
            return jsonify(new_model.dict())
        except IntegrityError:
            return jsonify({'error': 'The model name already exists'})
        except Exception as e:
            return jsonify({'error': str(e)})

    @staticmethod
    def get():
        try:
            if request.args.get('id', ''):
                model = TrainModel.get(model_id=request.args.get('id', 1))
                return jsonify(model.dict())
            else:
                models = TrainModel.query()
                all_models = [model.dict() for model in models]
                return jsonify(all_models)
        except NoResultFound:
            return jsonify({'error': 'The queried model does not exist'})
        except Exception as e:
            return jsonify({'error': str(e)})

    @staticmethod
    def put():
        try:
            body = request.json
            edited_model = TrainModel.edit(id=body['id'], serving=body['serving'])
            return jsonify(edited_model.dict())
        except NoResultFound:
            return jsonify({'error': 'The model does not exist'})
        except Exception as e:
            return jsonify({'error': str(e)})


api.add_resource(Prediction, "/predict")
api.add_resource(Training, "/train")

if __name__ == '__main__':
    app.run(debug=False)
