import tensorflow as tf
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
with tf.device('/gpu:0'):  # Replace with device you are interested in
  bytes_in_use = BytesInUse()
  b = tf.Variable(tf.zeros([10000, 10000]))
  a = tf.Variable(tf.zeros([30000, 30000]))

with tf.Session() as sess:
  print(sess.run(bytes_in_use))
  sess.run(tf.global_variables_initializer())
  print(sess.run(bytes_in_use))


"Server for running the model"

import json
import random

from flask import Flask, request
from flask_restful import Resource, Api, output_json

predict = lambda s: s

class UnicodeApi(Api):
    def __init__(self, *args, **kwargs):
        super(UnicodeApi, self).__init__(*args, **kwargs)
        self.app.config['RESTFUL_JSON'] = {
            'ensure_ascii': False
        }
        self.representations = {
            'application/json; charset=utf-8': output_json,
        }


app = Flask(__name__)
api = UnicodeApi(app)

class Dialog(Resource):
    def post(self):
        if request.json is None:
            return {'error': 'You should send me json data!'}, 400

        if not 'sentences' in request.json:
            return {'error': 'Your json request should inlcude `sentences`'}, 400

        try:
            # print(predict(request.json['sentences']))
            return {'result': predict(request.json['sentences'])}
        except Exception as e:
            print('Error occured:', e)

            return {'error': 'Something went wrong'}, 500



# api.add_resource(Style, '/style')
api.add_resource(Dialog, '/dialog')

if __name__ == '__main__':
    app.run(port=10101, host='0.0.0.0')