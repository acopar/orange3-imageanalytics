import random

from PIL.Image import open as open_image
from io import BytesIO
import numpy as np
import tensorflow as tf
import os
import string

# import asyncio
# from sanic.log import logger

MODEL_FILE = "inception_v3_batch.pb"


def preprocess_for_eval(images):
    images /= 255.0
    images -= 0.5
    images *= 2

    return images


def _import_tf_graph():
    with tf.gfile.FastGFile(MODEL_FILE, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def job_init():
    graph = tf.Graph()
    with graph.as_default():
        _import_tf_graph()
    return graph

def do(graph, jobs):
    with tf.Session(graph=graph) as session:
        images = []

        for i, (_, im) in enumerate(jobs):
            image = open_image(BytesIO(im))
            image = image.resize((299, 299))
            images.append(image)

        batch = np.stack(images).astype(float)
        batch = preprocess_for_eval(batch)

        output_t = session.graph.get_tensor_by_name("InceptionV3/Logits/Dropout_1b/Identity:0")
        input_t = session.graph.get_tensor_by_name("Placeholder:0")
        output = session.run(output_t, feed_dict={input_t: batch})

    output = output[:, 0, 0, :].tolist()
    keys = list(zip(*jobs))[0]
    return list(zip(keys, output))
