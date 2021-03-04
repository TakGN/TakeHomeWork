from flask_testing import TestCase

from web_server import app


class TestPrediction(TestCase):

    def create_app(self):
        return app

    def test_post(self):
        payload = {'email': 'takwa@gmail.com',
                   'model_name': 'wonderful_model'}
        response = app.test_client().post('/predict', data=payload,
                                          content_type='application/json')
        assert response.status_code == 200


class TestTraining(TestCase):

    def create_app(self):
        return app

    def test_post(self):
        payload = {"model_name": "FantasticModel",
                   "model_type": "GradientBoosting",
                   "model_params":  {
                       "model_params": {"n_estimators": 50, "learning_rate": 0.1},
                       "tf_idf_params": {
                           "ngram_range": [4, 5],
                           "strip_accents": "unicode",
                           "analyzer": "char",
                           "max_features": 1000}
                   }
                   }
        response = app.test_client().post('/train', data=payload,
                                          content_type='application/json')
        assert response.status_code == 200

    def test_get(self):
        response = app.test_client().get('/train', content_type='application/json')
        assert response.status_code == 200

    def test_put(self):
        payload = {"id": 1,
                   "serving": True}
        response = app.test_client().put('/train', data=payload,
                                         content_type='application/json')
        assert response.status_code == 200


