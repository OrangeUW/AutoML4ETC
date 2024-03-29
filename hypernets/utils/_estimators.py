import copy
import pickle

from sklearn.pipeline import Pipeline

from . import fs


def save_estimator(estimator, model_path):
    if isinstance(estimator, Pipeline) and hasattr(estimator.steps[-1][1], 'save') \
            and hasattr(estimator.steps[-1][1], 'load'):
        if fs.exists(model_path):
            fs.rm(model_path, recursive=True)
        fs.mkdirs(model_path, exist_ok=True)
        if not model_path.endswith(fs.sep):
            model_path = model_path + fs.sep

        stub = copy.copy(estimator)
        stub.steps[-1][1].save(f'{model_path}pipeline.model')
        with fs.open(f'{model_path}pipeline.pkl', 'wb') as f:
            pickle.dump(stub, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with fs.open(model_path, 'wb') as f:
            pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_estimator(model_path):
    model_path_ = model_path
    if not model_path_.endswith(fs.sep):
        model_path_ = model_path_ + fs.sep

    if fs.exists(f'{model_path_}pipeline.pkl'):
        with fs.open(f'{model_path_}pipeline.pkl', 'rb') as f:
            stub = pickle.load(f)
            assert isinstance(stub, Pipeline)

        estimator = stub.steps[-1][1]
        if fs.exists(f'{model_path_}pipeline.model') and hasattr(estimator, 'load'):
            est = estimator.load(f'{model_path_}pipeline.model')
            steps = stub.steps[:-1] + [(stub.steps[-1][0], est)]
            stub = Pipeline(steps)
    else:
        with fs.open(model_path, 'rb') as f:
            stub = pickle.load(f)

    return stub
