import sklearn.metrics as SM
import seaborn
from scipy.io import loadmat
from expiringdict import ExpiringDict
import re
import random
import pickle
import importlib
from pathlib import Path
import tensorflow.keras.models as KM
import tensorflow.keras.layers as KL
import tensorflow.keras.utils as KU
import tensorflow.keras.regularizers as KR
import tensorflow.keras.optimizers as KO
import tensorflow.keras.backend as KB
import tensorflow.keras.callbacks 
from hypernets.searchers.random_searcher import RandomSearcher
from hyperkeras.searchers.enas_rl_searcher import EnasSearcher
from hypernets.searchers.mcts_searcher import MCTSSearcher
from hyperkeras.search_space.enas_common_ops_dropoutCNN_05_1DCNN import *
from hyperkeras.layers_1DCNN_OPS import Input, Reshape
from hyperkeras.search_space.enas_layers_1D import FactorizedReduction
from hypernets.core.search_space import HyperSpace
import tensorflow as tf
import numpy as np
from hypernets.core.callbacks import SummaryCallback
from hypernets.core.ops import *
from hyperkeras.hyper_keras import HyperKeras
from hypernets.searchers.random_searcher import RandomSearcher
from hyperkeras.searchers.enas_rl_searcher import EnasSearcher
from hypernets.searchers.mcts_searcher import MCTSSearcher
import time
import numpy as N
import numpy.linalg as NL
import numpy.random as NR
import pandas as P
import matplotlib.pyplot as plt


def automl4etc_cnn_searchspace(input_shape, classes, arch='NR', init_filters=64, node_num=4, data_format=None,
                            classification_dropout=0,
                            hp_dict={}, use_input_placeholder=True,
                            weights_cache=None):
        
        space = HyperSpace()
        with space.as_default():
            if use_input_placeholder:
                input = Input(shape=input_shape, name='0_input')
                input_net = input
                input_net = Reshape(target_shape=(3 * 600, 1), name="0_input_Flatten")(input_net)
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
    
    def __init__(self):
        self.searcher = RandomSearcher
    
    def set_searcher(self, searcher="RS"):
        if searcher == "RL":
            self.searcher = EnasSearcher
        elif searcher == "RS":
            self.searcher = RandomSearcher
        elif searcher == "MCTS":
            self.searcher = MCTSSearcher
        return
    
    def search(self, train_dataset, test_dataset, input_shape, classes):
        searcher_space = self.searcher(
            lambda: automl4etc_cnn_searchspace(arch='NR', input_shape=(3, 600), classes=9),
            optimize_direction='max')
        hk = HyperKeras(searcher_space, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'],
                        callbacks=[SummaryCallback()])

        (x_train1, y_train1), (x_test1, y_test1) = tf.keras.datasets.mnist.load_data()

        # Rescale the images from [0,255] to the [0.0,1.0] range.
        x_train1, x_test1 = x_train1[..., np.newaxis] / 255.0, x_test1[..., np.newaxis] / 255.0
        y_train1 = tf.keras.utils.to_categorical(y_train1)
        y_test1 = tf.keras.utils.to_categorical(y_test1)

        hk.search(train_dataset, y_train1, test_dataset, y_test1, initial_lr=0.001, max_trials=200, epochs=40)
        assert hk.get_best_trial()
        
    
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

        print(get_class_counts( y_train ))
        print(get_class_counts( y_test ))

        train_dataset = tf.data.Dataset.from_tensor_slices( ( x_train, y_train ) )
        train_dataset = train_dataset.batch( 128 )

        test_dataset = tf.data.Dataset.from_tensor_slices( ( x_test, y_test ) )
        test_dataset = test_dataset.batch( 128 )

        return train_dataset, test_dataset


