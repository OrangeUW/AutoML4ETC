import scipy.io
import numpy
import tensorflow.keras.utils
import pickle
from expiringdict import ExpiringDict
from .predefs import MASKED_VAL

MASKED_VAL_SIZE = 0

class DataGenFlowonly(tensorflow.keras.utils.Sequence):
    """
    Generator class for feeding large data-set of giant PCAPs parsed
    and pre-processed into multiple MAT files to the model
    
    This class is similar to DataGenOrange but uses a much higher truncation value for the flow sequence.
    """
    
    def __init__(self, files, label_func, weight_func=None,
                 idxfilter=lambda x: True, \
                 batchsize=16, weights=None, \
                 n_flows=1000):
        """
        files: [str], contains a list of .mat files
        idxfilter: Filter the files by their index to generate training/test datasets.
        batchsize: Number of files (samples) for each batch
        weights: Either None, or an array of size NUM_TYPES for weighing the classes
        sparse: If false, categorical variables are encoded one-hot.
        header_truncate: If a positive number is provided, trim header to given size
        
        header_selector: If a function is provided, only one header packet will be selected based on header_selector
        """
        self.idxfilter = idxfilter
        self.files = files
        self.files = [f for idx,f in enumerate(self.files)if self.idxfilter(idx)]
        self.batchsize = batchsize
        self.cache = ExpiringDict(max_len=8, max_age_seconds=3600)
        self.weights = weights
        self.n_flows = n_flows
        
        self.label_func = label_func
        self.weight_func = weight_func
        
    def __len__(self):
        return self.int_ceildiv(len(self.files), self.batchsize)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache.get(idx)
        
        start_idx = idx * self.batchsize
        loaded = [x for x in self.files[start_idx:start_idx+self.batchsize]]
        loaded = [self.mat_readone(x) for x in loaded]
        loaded = [x for x in loaded if x]
        
        assert len(loaded) <= self.batchsize
        
        out_x1 = self.justified_merge([x[1] for x,y,w in loaded], MASKED_VAL_SIZE, self.n_flows)

        out_x = out_x1
        out_y = numpy.concatenate([y for x,y,w in loaded])
        out_w = numpy.concatenate([w for x,y,w in loaded])
        
        self.cache[idx] = (out_x, out_y, out_w)
        
        if self.weight_func:
            return out_x, out_y, out_w
        else:
            return out_x, out_y
    
    def justified_merge(self, li, mask, truncate=None):
        if not truncate:
            truncate = max([x[0].shape[1] for x in li])
        
        def padone(x):
            if truncate > x.shape[1]:
                pad = truncate - x.shape[1]
                return numpy.pad(x, ((0,0),(0,pad),(0,0)), 'constant', constant_values=mask)
            elif truncate < x.shape[1]:
                return x[:,:truncate,:]
            else:
                return x
        
        result = numpy.concatenate([padone(x) for x in li])
        assert result.shape[1] == truncate
        return result
        
    # Others
    def int_ceildiv(self, a: int, b: int):
        return -((-a) // b)

    def time_normaliser(self, t):
        return numpy.log(numpy.maximum(0, t) * 1000 + 1)
    def size_normaliser(self, s):
        # 128 being mean packet size
        return s / 128
    
    def domain_suffix(self, x):
        return '.'.join(x.split('.')[-2:])

    def mat_readone(self, f, verbose=False):
        loaded = scipy.io.loadmat(f)

        loaded_flow = loaded["flow"]
        loaded_label = loaded["label"][0]
        
        
        if verbose:
            print("File={}".format(f))
            print("Shape:  F={}".format(loaded_flow.shape))
            print("Flow={}, Label={}".format(loaded_flow, loaded_label))
            print("[0] F={}".format(loaded_flow))

        #assert loaded_header.shape[0] == loaded_flow.shape[0]

        # Normalise
        loaded_flow[:,0] = self.time_normaliser(loaded_flow[:,0])
        loaded_flow[:,1] = self.size_normaliser(loaded_flow[:,1])

        if verbose:
            print("Shapes: F={}".format(loaded_flow.shape))

        
        out_x = (None, numpy.array([loaded_flow]))
        out_y = self.label_func(loaded_label)
            
            
        if self.weight_func:
            out_w = self.weight_func(out_y)
        else:
            out_w = 1

        out_y = numpy.array([out_y])
        out_w = numpy.array([out_w])
        
        return out_x, out_y, out_w