import os
import pickle
from path import Path
import numpy as np
import numpy as N
import scipy.io
import time
import tensorflow


from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext


def get_spark(**kwargs):
    conf = SparkConf()
    conf.setAppName("Orange")
    conf.setMaster("local[64]")
    conf.set("spark.executor.memory", "1g")
    for k, v in kwargs.items():
        assert isinstance(k, str)
        assert isinstance(v, str)
        conf.set(k, v)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    return spark, sc, sqlContext


# deprecated
def load_dataset_superflow(ensemble, reload_aux=False, path_base='/home/orange/dataset_sept/orange_mat'):

    pathBase = Path(path_base)
    def datapath_to_mat(datapath):
        return str(pathBase / (datapath + '.mat'))

    with open(pathBase / 'metadata.pickle', 'rb') as f:
        metadata = pickle.load(f)

    # Set RELOAD_ALL to true to update the cache. It is very handy when this notebook crashes
    RELOAD_ALL = reload_aux
    if RELOAD_ALL:
        spark, sc, _ = get_spark()
        # Read file containing all streams for which CIC can extract at least 1 flow.
        with open(pathBase / 'all_cfm.txt', 'r') as f:
            datapath_all = [x.strip() for x in f]
            print(f"{len(datapath_all)} streams extracted")

        with open(pathBase / 'flowgroups.pickle', 'rb') as f:
            flowgroups = pickle.load(f)
            print(f"{len(flowgroups)} flow groups extracted")
            
        # Read CIC flow meter matrix
        cfm = scipy.io.loadmat(pathBase / 'cfm.mat')
        cfm.keys()
        
        # Construct the auxiliary map.
        usable_cols = cfm['usable_cols'][0]
        auxiliary_map = ((cfm['flows'] - cfm['mu'])/cfm['sigma'])[:,usable_cols]
        assert np.isfinite(auxiliary_map).all()
        auxiliary_map = {k:auxiliary_map[i] for i,k in enumerate(datapath_all)}
        
        # Filter datapath_all
        def sublabeling_func(x):
            x = datapath_to_mat(x)
            if not os.path.exists(x):
                return -2
            return ensemble.label_func(scipy.io.loadmat(x)['label'][0])
        labels = sc.parallelize(datapath_all) \
            .map(sublabeling_func)\
            .collect()
        datapath_all = [x for i,x in enumerate(datapath_all) if labels[i] >= 0]
        ensemble.generate_counts(labels)
        print("Ensemble counts: {}".format(ensemble.counts))
        
        
        labels = sc.parallelize(datapath_all) \
            .map(lambda x:scipy.io.loadmat(datapath_to_mat(x))['label'][0]) \
            .map(ensemble.label_func) \
            .collect()
        ensemble.generate_counts(labels)
        print(ensemble.counts)
        
        
        auxiliary_map = {k:auxiliary_map[k] for k in datapath_all}
        
        dsall_set = set(datapath_all)
        flowgroups = sc.parallelize(flowgroups) \
            .filter(lambda x:all(y['datapath'] in datapath_all for y in x)) \
            .collect()
        print(f"{len(flowgroups)} groups collected")
        
        Path(pathBase / 'training').mkdir(parents=True, exist_ok=True)
        with open(pathBase / 'training/datapath_all.txt', 'w') as f:
            for x in datapath_all:
                f.write(x + '\n')
        with open(pathBase / 'training/flowgroups.pickle', 'wb') as f:
            pickle.dump(flowgroups, f)
            
        scipy.io.savemat(pathBase / 'training/auxiliary_map.mat', {
            'aux_map': np.stack([auxiliary_map[k] for k in datapath_all]),
            'labels': np.array(labels),
        })
    else:
        with open(pathBase / 'training/datapath_all.txt', 'r') as f:
            datapath_all = [x.strip() for x in f]
            print(f"{len(datapath_all)} streams extracted")
        with open(pathBase / 'training/flowgroups.pickle', 'rb') as f:
            flowgroups = pickle.load(f)
        auxiliary_map = scipy.io.loadmat(pathBase / 'training/auxiliary_map.mat')
        auxiliary_map, labels = auxiliary_map['aux_map'], auxiliary_map['labels']
        AUXILIARY_SIZE = auxiliary_map.shape[1]
        print("Auxiliary dimensions: {}".format(AUXILIARY_SIZE))
        assert auxiliary_map.shape[0] == len(datapath_all)
        auxiliary_map = {k:auxiliary_map[i] for i,k in enumerate(datapath_all)}
        ensemble.generate_counts(labels[0])
        print("Ensemble counts: {}".format(ensemble.counts))

    return {
        "datapath_all": datapath_all,
        "flowgroups": flowgroups,
        "auxiliary_map": auxiliary_map,
        "metadata": metadata
    }


def load_dataset(ensemble, reload=False, superflow=False, path_base="/home/orange/dataset_sept/orange_mat"):
    
    pathBase = Path(path_base)
    
    with open(pathBase / 'metadata.pickle', 'rb') as f:
        metadata = pickle.load(f)
    
    if reload:

        # Read file containing all streams for which CIC can extract at least 1 flow.
        with open(pathBase / 'all_cfm.txt', 'r') as f:
            datapath_all = [x.strip() for x in f]
            print(f"{len(datapath_all)} total flows in dataset")

        with open(pathBase / 'flowgroups.pickle', 'rb') as f:
            flowgroups_all = pickle.load(f)
            print(f"{len(flowgroups_all)} total flow-groups in dataset")

        # Read CIC flow meter matrix
        cfm = scipy.io.loadmat(pathBase / 'cfm.mat')
        
        
        mat = cfm["flows"]
        mat_safe = np.nan_to_num(mat, nan=0, neginf=0, posinf=0)

        # normalize
        mu = np.mean(mat_safe, axis=0)
        sigma = np.std(mat_safe, axis=0)
        mat_norm = (mat_safe - mu) / sigma
        mat_norm = np.nan_to_num(mat_norm, nan=0)

        # replace values of entries that had +inf,-inf,NaN in original matrix
        nan_val = int(-1)
        pos_inf_val = int(+2)
        neg_inf_val = int(-2)
        mat_norm[mat == np.inf] = pos_inf_val
        mat_norm[mat == -np.inf] = neg_inf_val
        mat_norm[np.isnan(mat)] = nan_val

        # final values
        auxiliary_map = mat_norm
        auxiliary_size = auxiliary_map[0].shape[0]

        # sanity checks (no +inf,-inf,NaN in the final result)
        assert not np.isnan(auxiliary_map).any()
        assert np.isfinite(auxiliary_map).all()
            
        auxiliary_map = {k:auxiliary_map[i] for i,k in enumerate(datapath_all)}

        # Filter datapath_all
        def sublabeling_func(x):
            """
            Returns label for the given relative datapath

            Input:
                x - str: relative datapath

            Returns:
                int: label (-1 if lookup fails, -2 if load fails)
            """
            x = datapath_to_mat(x)
            if not os.path.exists(x):
                return -2
            return ensemble.label_func(scipy.io.loadmat(x)['label'][0])

        def datapath_to_mat(datapath):
            """
            Convert relative datapath to absolute MAT file path
            """
            return str(pathBase / (datapath + '.mat')) 

        def label_superflow(fg):
            for f in fg:
                label = sublabeling_func(f["datapath"])
                if label >= 0:
                    return label
        
        spark, sc, _ = get_spark()

        labelable_flows = sc.parallelize(datapath_all) \
            .map(lambda x: (x, sublabeling_func(x)))\
            .filter(lambda x: x[1] >= 0) \
            .collect()

        new_samples = [x[0] for x in labelable_flows]
        new_labels = [x[1] for x in labelable_flows]
        
        ensemble.generate_counts(new_labels)
        print("Ensemble counts: {}".format(ensemble.counts))
        print(f"{len(new_samples)} usable (labeled) flows collected")
        
        dsall_set = set(datapath_all)
        dsgood_set = set(new_samples)
        flowgroups = sc.parallelize(flowgroups_all) \
            .filter(lambda x:all(y['datapath'] in dsall_set for y in x) and any(y['datapath'] in dsgood_set for y in x)) \
            .collect()
        
        print(f"{len(flowgroups)} usable (labeled) groups collected")

        if superflow:
            fg_labels = sc.parallelize(flowgroups).map(label_superflow).collect()
            ensemble.generate_counts(fg_labels)
            print("Ensemble counts updated with flowgroup labels: {}".format(ensemble.counts))
        
        Path(pathBase / 'training').mkdir_p()
        with open(pathBase / 'training/datapath_all.txt', 'w') as f:
            for x in datapath_all:
                f.write(x + '\n')
        with open(pathBase / 'training/samples.txt', 'w') as f:
            for x in new_samples:
                f.write(x + '\n')
        with open(pathBase / 'training/flowgroups.pickle', 'wb') as f:
            pickle.dump(flowgroups, f)

        if superflow:
            with open(pathBase / 'training/fg_labels.pickle', 'wb') as f:
                pickle.dump(fg_labels, f)

        scipy.io.savemat(pathBase / 'training/auxiliary_map.mat', {
            'aux_map': N.stack([auxiliary_map[k] for k in datapath_all]),
            'labels': N.array(new_labels),
        })
        
    else:
        
        with open(pathBase / 'training/datapath_all.txt', 'r') as f:
            datapath_all = [x.strip() for x in f]
            print(f"{len(datapath_all)} streams extracted")
            
        with open(pathBase / 'training/samples.txt', 'r') as f:
            new_samples = [x.strip() for x in f]
            print(f"{len(new_samples)} labeled streams available")
            
        with open(pathBase / 'training/flowgroups.pickle', 'rb') as f:
            flowgroups = pickle.load(f)
            print(f"{len(flowgroups)} flow groups extracted")
            
        auxiliary_map = scipy.io.loadmat(pathBase / 'training/auxiliary_map.mat')
        auxiliary_map, labels = auxiliary_map['aux_map'], auxiliary_map['labels']
        AUXILIARY_SIZE = auxiliary_map.shape[1]
        print("Auxiliary dimensions: {}".format(AUXILIARY_SIZE))
        assert auxiliary_map.shape[0] == len(datapath_all)
        auxiliary_map = {k:auxiliary_map[i] for i,k in enumerate(datapath_all)}
        ensemble.generate_counts(labels[0])

        if superflow:
            with open(pathBase / 'training/fg_labels.pickle', 'rb') as f:
                fg_labels = pickle.load(f)
            ensemble.generate_counts(fg_labels)
        
        print("Ensemble counts: {}".format(ensemble.counts))
        print(f"{len(new_samples)} usable (labeled) flows collected")
    
    if superflow:
        pass

    return {
        "datapath_all": datapath_all,
        "samples": new_samples,
        "flowgroups": flowgroups,
        "auxiliary_map": auxiliary_map,
        "metadata": metadata,
    }


def callback_checkpoints(model, base_path="Models"):
    checkpoint_path =  base_path + "/%s/training_%d/cp-{epoch:04d}.ckpt" % (model.name, int(time.time()))
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    print("Output path:", checkpoint_path)
    return cp_callback


def merge_history_objects(*args):
    history_full = {'loss': [], 'val_loss': [], 'val_sparse_categorical_accuracy': [], 'sparse_categorical_accuracy': [] }
    for x in args:
        for k in x.history:
            history_full[k] += x.history[k] 
    return history_full



