# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.model.hyper_model import HyperModel
from hypernets.model.estimator import Estimator

import json
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras import utils
import tensorflow as tf
import numpy as np
import gc
from .layer_weights_cache import LayerWeightsCache
import logging

logger = logging.getLogger(__name__)


class KerasEstimator(Estimator):
    def __init__(self, space_sample, optimizer, loss, metrics, max_model_size=0, weights_cache=None,
                 visualization=False):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.max_model_size = max_model_size
        self.weights_cache = weights_cache
        self.visualization = visualization
        self.model = self._build_model(space_sample)
        Estimator.__init__(self, space_sample=space_sample)

    def _build_model(self, space_sample):
        K.clear_session()
        gc.collect()
        space_sample.weights_cache = self.weights_cache
        model = space_sample.keras_model(deepcopy=self.weights_cache is None)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        if self.max_model_size > 0:
            model_size = compute_params_count(model)
            assert model_size <= self.max_model_size, f'Model size out of limit:{self.max_model_size}'
        if self.visualization:
            utils.plot_model(model, f'model_{space_sample.space_id}.png', show_shapes=True)
        return model

    def summary(self):
        self.model.summary()

    def fit(self, X, y, validation_gen = None, initial_lr=0.0001, lr_exp_rate=0.5, lr_exp_epoch=10.0, class_weight=None, **kwargs):
        import math
        from tensorflow.keras.callbacks import LearningRateScheduler

        initial_learning_rate = initial_lr #for the time-series data --> use 0.001 for the header data
        
        def lr_exp_decay(epoch, lr):
#             k = 0.1
#             if epoch < 10:
#                 return initial_learning_rate
#             else:
#                 return initial_learning_rate * math.exp(-k*epoch)
            drop_rate = lr_exp_rate
            epochs_drop = lr_exp_epoch
            return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))
        
        #self.model.fit(X, y, **kwargs)
        
        
        if not validation_gen:
            self.model.fit(X, callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1)], class_weight=class_weight, **kwargs)
        else:
            self.model.fit(X, callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1)], class_weight=class_weight, validation_freq=10, validation_data=validation_gen, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, metrics=None, **kwargs):
        scores = self.model.evaluate(X) #removed kwargs -> because epoch is not supported and also meaningle
        result = {k: v for k, v in zip(self.model.metrics_names, scores)}
        return result

    def save(self, model_file, to_save=False):
        #save_model(self.model, model_file, save_format='h5')
        if to_save:
            self.model.save("ENAS_models/{}".format(model_file))
            print("Model was saved at: ENAS_models/{}".format(model_file))
#             json_model = self.model.to_json()
#             with open("ENAS_models/{}.json".format(model_file), 'w') as json_file:
#                 json_file.write(json_model)
#                 print("Model was saved at: ENAS_models/{}.json".format(model_file))
        else:
            pass


class HyperKeras(HyperModel):
    def __init__(self, searcher, optimizer, loss, metrics, dispatcher=None, callbacks=[],
                 reward_metric=None, max_model_size=0, one_shot_mode=False, one_shot_train_sampler=None,
                 visualization=False, task=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.max_model_size = max_model_size
        if reward_metric is None:
            reward_metric = metrics[0]
        if one_shot_mode:
            self.weights_cache = LayerWeightsCache()
        else:
            self.weights_cache = None
        self.one_shot_mode = one_shot_mode
        self.one_shot_train_sampler = one_shot_train_sampler if one_shot_train_sampler is not None else searcher
        self.visualization = visualization
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric,
                            task=task)

    def _get_estimator(self, space_sample):
        estimator = KerasEstimator(space_sample, optimizer=self.optimizer, loss=self.loss, metrics=self.metrics,
                                   max_model_size=self.max_model_size, weights_cache=self.weights_cache,
                                   visualization=self.visualization)
        return estimator

    def build_dataset_iter(self, X, y, batch_size=32, buffer_size=None, reshuffle_each_iteration=None, repeat_count=1):
        if buffer_size is None:
            buffer_size = len(X[-1])
        dataset = tf.data.Dataset. \
            from_tensor_slices((X, y)). \
            shuffle(buffer_size=buffer_size, reshuffle_each_iteration=reshuffle_each_iteration). \
            repeat(repeat_count). \
            batch(batch_size)

        return iter(dataset)

    def fit_one_shot_model_epoch(self, X, y, batch_size=32, steps=None, epoch=0):
        step = 0
        dataset_iter = self.build_dataset_iter(X, y, batch_size=batch_size)
        for X_batch, y_batch in dataset_iter:
            sample = self.one_shot_train_sampler.sample()
            est = self._get_estimator(space_sample=sample)
            est.fit(X_batch, y_batch, batch_size=batch_size, epochs=1)
            step += 1
            print(f'One-shot model training, Epoch[{epoch}], Step[{step}]')
            if steps is not None and step >= steps:
                break
        print(f'One-shot model training finished. Epoch[{epoch}], Step[{step}]')

    def export_trial_configuration(self, trial):
        return None


def compute_params_count(model):
    assert model.built, ''
    return int(np.sum([K.count_params(weights) for weights in model.trainable_weights]))
