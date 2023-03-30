from pathlib import Path
from hypernets.searchers.random_searcher import RandomSearcher
from hyperkeras.searchers.enas_rl_searcher import EnasSearcher
from hypernets.searchers.mcts_searcher import MCTSSearcher
from hyperkeras.search_space.enas_common_ops_dropoutCNN_05_1DCNN import *
from hyperkeras.layers_1DCNN_OPS import Input, Reshape
from hyperkeras.search_space.enas_layers_1D import FactorizedReduction
from hypernets.core.search_space import HyperSpace
from hypernets.core.callbacks import SummaryCallback
from hypernets.core.ops import *
from hyperkeras.hyper_keras import HyperKeras
from hypernets.searchers.random_searcher import RandomSearcher
from hyperkeras.searchers.enas_rl_searcher import EnasSearcher
from hypernets.searchers.mcts_searcher import MCTSSearcher
import re
import tensorflow as tf
import yaml
import numpy as np

import random
import pickle
import commonio.datagen_separated

_CONF = None





def _get_conf_dict():
    global _CONF

    if _CONF:
        return dict(_CONF)

    with open(Path(__file__).resolve().parent / "config.yml") as f:
        _CONF = yaml.safe_load(f)
        return dict(_CONF)

def get_conf(key, default=None):
    return _get_conf_dict().get(key, default) if default is not None else _get_conf_dict()[key]

arch = get_conf("searchspace.arch", "NR")
init_filters=get_conf("searchspace.init_filters", 64)
node_num=get_conf("searchspace.node_num", 4)
optimize_direction= get_conf("search.optimize_direction", "max")
optimizer = get_conf("search.optimizer", "adam")
loss = get_conf("search.loss", "sparse_categorical_crossentropy")
metrics = get_conf("search.metrics", ['sparse_categorical_accuracy'])
searcher = get_conf("search.searchalgo", "RS")
initial_lr=get_conf("search.initial_learning_rate", 0.001)
lr_exp_rate=get_conf("search.learning_rate_decline_cut", 0.5)
lr_exp_epoch=get_conf("search.learning_rate_decline_every_epoch", 10)
max_trials=get_conf("search.max_trials", 100)
epochs=get_conf("search.training_epoch_per_trial", 40)




def automl4etc_cnn_searchspace(input_shape, classes, arch='NR', init_filters=64, node_num=4, data_format=None,
                            classification_dropout=0,
                            hp_dict={}, use_input_placeholder=True,
                            weights_cache=None):
        
        space = HyperSpace()
        with space.as_default():
            if use_input_placeholder:
                input = Input(shape=input_shape, name='0_input')
                input_net = input
                input_net = Reshape(target_shape=(input_shape[0]*input_shape[1], 1), name="0_input_Flatten")(input_net)
            else:
                input = None
                input_net = input
            stem, input_net = stem_op(input_net, init_filters, data_format)
            node0 = stem
            node1 = stem
            reduction_no = 0
            normal_no = 0

            for l in arch:
                if l == 'N':
                    normal_no += 1
                    type = 'normal'
                    cell_no = normal_no
                    is_reduction = False
                else:
                    reduction_no += 1
                    type = 'reduction'
                    cell_no = reduction_no
                    is_reduction = True
                filters = (2 ** reduction_no) * init_filters

                if is_reduction:
                    node0 = FactorizedReduction(filters, f'{normal_no + reduction_no}_{type}_C{cell_no}_0', data_format)(
                        node0)
                    node1 = FactorizedReduction(filters, f'{normal_no + reduction_no}_{type}_C{cell_no}_1', data_format)(
                        node1)
                x = conv_layer(hp_dict, f'{normal_no + reduction_no}_{type}', cell_no, [node0, node1], filters, node_num,
                               is_reduction)
                node0 = node1
                node1 = x
            logit = classification(x, classes, classification_dropout, data_format)
            space.set_inputs(input)
            if weights_cache is not None:
                space.weights_cache = weights_cache

        return space
    

class automl4etc():
        
    def search(self, train_dataset, test_dataset, input_shape, classes):
        global arch, init_filters, node_num, optimize_direction
        global optimizer, loss, metrics, searcher, initial_lr
        global lr_exp_rate, lr_exp_epoch, max_trials, epochs
        
        searcher = "RS"
        if searcher == "RL":
            searcher = EnasSearcher
        elif searcher == "RS":
            searcher = RandomSearcher
        elif searcher == "MCTS":
            searcher = MCTSSearcher
        else:
            print("***ERROR, invalid searcher: {}*** reverting to RS".format(searcher))
            searcher = RandomSearcher
        
        searcher_space = searcher(
            lambda: automl4etc_cnn_searchspace(arch=arch, input_shape=input_shape, classes=classes,
            hp_dict={},                                   
            init_filters=init_filters,
            node_num=node_num),
            optimize_direction= optimize_direction)
        hk = HyperKeras(searcher_space, optimizer=optimizer, loss=loss, metrics=metrics,
                        callbacks=[SummaryCallback()])

        (x_train1, y_train1), (x_test1, y_test1) = tf.keras.datasets.mnist.load_data()

        # Rescale the images from [0,255] to the [0.0,1.0] range.
        x_train1, x_test1 = x_train1[..., np.newaxis] / 255.0, x_test1[..., np.newaxis] / 255.0
        y_train1 = tf.keras.utils.to_categorical(y_train1)
        y_test1 = tf.keras.utils.to_categorical(y_test1)
        


        hk.search(train_dataset, y_train1, test_dataset, y_test1,
                  initial_lr=initial_lr,
                  lr_exp_rate=lr_exp_rate,
                  lr_exp_epoch=lr_exp_epoch,
                  max_trials=max_trials, 
                  epochs=epochs)
        assert hk.get_best_trial()
        
    def quic_ucdavis_data_loader(self, path='./quic-dataset'):
        NAMES = ["GoogleDoc", "GoogleDrive", "GoogleMusic", "GoogleSearch", "Youtube"]
        def label_func(x):
            return NAMES.index(x)

        with open(path+"/full.pickle", "rb") as f:
            samples = pickle.load(f)
            random.seed(3549)
            random.shuffle(samples)

        samples = [samples_address.replace('/home/orange/dataset-mat/quic/', path+'/quic-data/') for samples_address in samples]



        kwargs = {
            "batchsize":8,
            'n_flows': 1024,
            'flow_separate_features': True,
            'flow_only': True,
        #     'stft': {'nperseg': 32, 'noverlap': 30, 'mode': 'rainbow'},
            'path_transform_func': lambda x:x
        }

        train_gen = commonio.datagen_separated.DataGenSeparated(samples, label_func, idxfilter=lambda x:(x%5!=2), **kwargs)
        val_gen   = commonio.datagen_separated.DataGenSeparated(samples, label_func, idxfilter=lambda x:(x%5==2), **kwargs)

        return train_gen, val_gen
    def mldit_data_loader(self, path='MLDIT_flow_headers_with_sni.txt'):
        import json

        tls_json = None

        with open(path, 'r' ) as f:
            tls_json = json.load( f )

        file_paths = list( tls_json.keys() )
        len( file_paths )

        file_paths_to_use = []
        packet_count = dict()

        for f in file_paths:
            num_of_packets = len( tls_json[ f ] )
            if num_of_packets > 0:
                file_paths_to_use.append( f )
            if num_of_packets not in packet_count:
                packet_count[ num_of_packets ] = 0
            packet_count[ num_of_packets ] += 1

        print( "Num of flows with at least 1 packet: ", len( file_paths_to_use ) )
        sorted( packet_count.items() )

        

        def get_label_hierarchy( label : str ):
            def find_from_regex( regex_str : str ):
                results = re.findall( regex_str, label )
                if len( results ) == 0:
                    return None
                results = results[ 0 ].split( '_' )
                if len( results ) != 2:
                    return None
                return results[ 1 ]

            APP_CAT_REG = 'AppCat[^_]+_[^-]+'
            APP_PROT_REG = 'AppProt[^_]+_[^-]+'
            NAV_REG = 'Nav[^_]+_[^-]+'
            OP_REG = 'Ope_[^-]+'

            results = dict()
            results[ 'service' ] = find_from_regex( APP_CAT_REG )
            results[ 'app' ] = find_from_regex( APP_PROT_REG )
            results[ 'nav' ] = find_from_regex( NAV_REG )
            results[ 'op' ] = find_from_regex( OP_REG )

            return results

        app_category_buckets = dict()

        for f in file_paths_to_use:
            app_category = get_label_hierarchy( f )[ 'service' ]
            if app_category not in app_category_buckets:
                app_category_buckets[ app_category ] = []
            app_category_buckets[ app_category ].append( f )

        app_category_buckets.keys()
        app_categories_sorted = sorted( list( app_category_buckets.keys() ) )

        for c in app_categories_sorted:
            print( c + ':', len( app_category_buckets[ c ] ) )

        import numpy

        def get_json_to_bytes_transformer( packet_cutoff : int, byte_cutoff : int, mask_sni : bool, normalize : bool, expand_dims : bool ):
            def get_sni_from_packet( packet_dict ):
                SNI_KEY = 'tls.handshake.extensions_server_name'
                if SNI_KEY not in packet_dict:
                    return None
                result = packet_dict[ SNI_KEY ]
                assert( len( result ) == 1 )
                sni_val = result[ 0 ]
                return sni_val

            def get_ascii_str_as_hex( ascii_str : str ):
                return ''.join( [ hex( int( ord( c ) ) )[ 2: ] for c in ascii_str ] )

            def hex_str_to_byte_ints( hex_str : str ):
                assert( len( hex_str ) % 2 == 0 )
                byte_ints = []
                for i in range( len( hex_str ) // 2 ):
                    start_index = i * 2
                    val = int( hex_str[ start_index : start_index + 2 ], 16 )
                    byte_ints.append( val )
                return byte_ints

            def transform_func( json_dict ):
                packet_dicts = []

                for i, d in enumerate( json_dict ):
                    if i >= packet_cutoff:
                        break
                    packet_dicts.append( d[ '_source' ][ 'layers' ] )

                TCP_PAYLOAD = 'tcp.payload'
                transformed_data = numpy.zeros( ( packet_cutoff, byte_cutoff ) )
                for i, p in enumerate( packet_dicts ):
                    byte_str = p[ TCP_PAYLOAD ][ 0 ]
                    assert( len( byte_str ) % 2 == 0 )

                    sni_str = get_sni_from_packet( p )
                    if sni_str is not None:
                        sni_str_to_hex = get_ascii_str_as_hex( sni_str )
                        assert( byte_str.find( sni_str_to_hex ) != -1 )
                        replacement_str = '0' * len( sni_str_to_hex )
                        byte_str = byte_str.replace( sni_str_to_hex, replacement_str )

                    byte_ints = hex_str_to_byte_ints( byte_str )
                    byte_ints = byte_ints[ :byte_cutoff ]
                    transformed_data[ i, :len( byte_ints ) ] = byte_ints

                if normalize:
                    transformed_data /= 255.0

                if expand_dims:
                    transformed_data = numpy.expand_dims( transformed_data, axis=-1 )

                return transformed_data       

            return transform_func

        def get_file_path_to_bytes_transformer( packet_cutoff : int, byte_cutoff : int, mask_sni : bool, normalize : bool, expand_dims : bool ):
            json_to_bytes_transformer = get_json_to_bytes_transformer( packet_cutoff, byte_cutoff, mask_sni, normalize, expand_dims )

            def transform_func( file_path : str ):
                json_dict = tls_json[ file_path ]
                return json_to_bytes_transformer( json_dict )

            return transform_func

        x_all = []
        y_all = []

        transform_func = get_file_path_to_bytes_transformer( 3, 600, True, True, False )

        for i, c in enumerate( app_categories_sorted ):
            print( "Transforming data for", c )
            for j, f in enumerate( app_category_buckets[ c ] ):
                if j % 100 == 0:
                    print( j, 'out of', len( app_category_buckets[ c ] ) )
                data = transform_func( f )
                label = i
                x_all.append( data )
                y_all.append( label )

        print( numpy.shape( numpy.asarray( x_all ) ) )
        print( numpy.shape( numpy.asarray( y_all ) ) )

        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split( x_all, y_all, stratify=y_all, test_size=0.2 )

        def get_class_counts( labels ):
            class_counts = dict()
            for l in labels:
                class_val = app_categories_sorted[ l ]
                if class_val not in class_counts:
                    class_counts[ class_val ] = 0
                class_counts[ class_val ] += 1
            assert( len( class_counts ) == len( app_categories_sorted ) )
            print( sorted( class_counts.items() ) )
            return class_counts
        import tensorflow as tf
        print(get_class_counts( y_train ))
        print(get_class_counts( y_test ))

        train_dataset = tf.data.Dataset.from_tensor_slices( ( x_train, y_train ) )
        train_dataset = train_dataset.batch( 128 )

        test_dataset = tf.data.Dataset.from_tensor_slices( ( x_test, y_test ) )
        test_dataset = test_dataset.batch( 128 )

        return train_dataset, test_dataset


