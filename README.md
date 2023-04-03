# AutoML4ETC
AutoML4ETC: Automated Neural Architecture Search for Encrypted Traffic Classification on Orange Mobile Network
Navid Malekghaini*, Elham Akbari Azirani*, Mohammad A. Salahuddin*, Noura Limam*, Raouf Boutaba*
Bertrand Mathieu† , Stephanie Moteau† , and Stephane Tuffin†
*University of Waterloo, Canada, † Orange Labs, France

<img alt="orange" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Orange_logo.svg/766px-Orange_logo.svg.png" width="80" /> <img src="https://dataverse.scholarsportal.info/logos/41143/Waterloo.png" width="240" alt="uwaterloo"/>


## About
We propose AutoML4ETC, a novel tool to automatically design efficient and high-performing neural architectures for encrypted traffic classification. We define a novel, powerful search space tailored specifically for the near real-time classification of encrypted traffic using packet header bytes. We show that with different search strategies over our search space, AutoML4ETC generates neural architectures that outperform the state-of-the-art encrypted traffic classifiers on several benchmark datasets, including real-world datasets of TLS and QUIC traffic collected from the Orange mobile network. In addition to being more accurate, AutoML4ETC’s architectures are significantly more efficient and lighter in terms of the number of parameters. For the reproducibility of results, we further showcase the superior performance of AutoML4ETC in comparison to state-of-the-art NAS techniques on a publicly available QUIC dataset. Finally, we make AutoML4ETC publicly available.

## Installation and use


```bash
./setup_automl4etc.sh
./activate_notebooksrv.sh
```

After installation you can browse to your localhost for accessing the jupyter notebook that can run AutoML4ETC library.


## Sample notebook and sample model generated

run sample notebook: test_automl4etc_ucDavisQUIC.ipynb

![(sample model picture should be here)](https://github.com/OrangeUW/AutoML4ETC/blob/main/Discovered_model.png?raw=true)

## The main API's to use
```python
#Using GPU libraries in the Tensorflow
import os 
import tensorflow

#if you have another GPU to use (eg. multiple GPUs, set 0 to the desired GPU number)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if tensorflow.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

#import AutoML4ETC
import automl4etc_common

#print current configuration
print(automl4etc_common._get_conf_dict())

automl = automl4etc_common.automl4etc()
train, test = automl.quic_ucdavis_data_loader(path='./quic-dataset')
automl.search(train_dataset=train, test_dataset=test, input_shape=(1024, 3), classes=5) #train_dataset and test_dataset inputs are A generator or keras.utils.Sequence and classes is the number of classes in the output

#now you can load saved models with the Keras API, this is the sample address in the sample notebook used
from keras.models import load_model

model = load_model('ENAS_models/00003_aef4978c-ce98-11ed-9fa3-e725897beba4') #replace with 'ENAS_models/path_to_model'

#If you like to visualize the model which AutoML4ETC discovered you can use Keras API
from keras.utils import plot_model

plot_model(model)

```

## Default configurations

```yaml
searchspace.arch: 'NR'  # this could be 'NR', 'NRNR' or 'NRNRNR', more 'N'ormal cells or 'R'eduction cells is not recommended
searchspace.init_filters: 64 # number of initial filters to begin process
searchspace.node_num: 4 # number of nodes per cell
searchspace.classification_dropout: 0 # final dropout rate before softmax layer

search.searchalgo: 'RL' # search algorithm -> 'RL', 'RS', 'MCTS'
search.loss: 'sparse_categorical_crossentropy' # kind of loss for the model evaluation, supported: https://keras.io/api/losses/
search.metrics: ['sparse_categorical_accuracy'] # array of metrics to monitor, supported: https://keras.io/api/metrics/
search.optimizer: 'adam' # optimizer to use, supported: https://keras.io/api/optimizers/
search.optimize_direction: 'max' #'max' or 'min' -> for the search.metric chosen
search.initial_learning_rate: 0.001 # initial learning rate
search.learning_rate_decline_cut: 0.5 # how much to cut (ie. decline) the learning rate every 10 (default) epochs
search.learning_rate_decline_every_epoch: 10 # wether cut in half every 10 (default) epochs or not
search.max_trials: 100 # maximum trials for searching time
search.training_epoch_per_trial: 40 # maximum epochs for training the child model per trial
```


