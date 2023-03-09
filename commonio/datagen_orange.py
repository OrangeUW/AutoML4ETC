import scipy.io
import numpy
import tensorflow.keras.utils
import pickle
from expiringdict import ExpiringDict
from .predefs import MASKED_VAL

TRUNCATE_SIZE = 500



class DataGenOrange(tensorflow.keras.utils.Sequence):
    """
    Generator class for feeding large data-set of giant PCAPs parsed
    and pre-processed into multiple MAT files to the model
    """
    
    def __init__(self, files, domains, idxfilter=lambda x: True, \
                 batchsize=16, weights=None, sparse=True, \
                 truncate=0, multi_input=True):
        """
        files: [str], contains a list of .mat files
        idxfilter: Filter the files by their index to generate training/test datasets.
        batchsize: Number of files (samples) for each batch
        weights: Either None, or an array of size NUM_TYPES for weighing the classes
        sparse: If false, categorical variables are encoded one-hot.
        truncate: If a positive number is provided, all inputs are trimmed to this size.
        multi_input: If false, the input packet and flow streams are combined.
        """
        self.idxfilter = idxfilter
        self.files = files
        self.files = [f for idx,f in enumerate(self.files)if self.idxfilter(idx)]
        self.batchsize = batchsize
        self.cache = ExpiringDict(max_len=8, max_age_seconds=3600)
        self.weights = weights
        self.sparse = sparse
        self.truncate = truncate
        self.multi_input = multi_input
        
        self.domains = domains
    
    def __len__(self):
        return self.int_ceildiv(len(self.files), self.batchsize)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache.get(idx)
        
        assert hasattr(self, 'truncate')
        
        start_idx = idx * self.batchsize
        loaded = [x for x in self.files[start_idx:start_idx+self.batchsize]]
        loaded = [self.mat_readone(x, sparse=self.sparse) for x in loaded]
        loaded = [x for x in loaded if x]
        
        assert len(loaded) <= self.batchsize
        
        if self.multi_input:
            if self.truncate:
                # Find max length sequence
                l = self.truncate
            else:
                l = max([x[0].shape[1] for x,y,l in loaded])

            def padone(x):
                if l > x.shape[1]:
                    pad = l - x.shape[1]
                    return numpy.pad(x, ((0,0),(0,pad),(0,0)), 'constant', constant_values=MASKED_VAL)
                elif l < x.shape[1]:
                    assert self.truncate
                    return x[:,:l,:]
                else:
                    return x


            #for x in loaded:
            #    print("X={}, Y={}".format(x[0].shape, x[1].shape))

            out_x0 = numpy.concatenate([padone(x[0]) for x,y,l in loaded])
            out_x1 = numpy.concatenate([padone(x[1]) for x,y,l in loaded])
            assert out_x0.shape[0] == out_x1.shape[0]
            assert out_x0.shape[1] == out_x1.shape[1]
            out_x = [out_x0, out_x1]
        else:
            if self.truncate:
                # Find max length sequence
                l = self.truncate
            else:
                l = max([x.shape[1] for x,y,l in loaded])

            def padone(x):
                if l > x.shape[1]:
                    pad = l - x.shape[1]
                    return numpy.pad(x, ((0,0),(0,pad),(0,0)), 'constant', constant_values=MASKED_VAL)
                elif l < x.shape[1]:
                    assert self.truncate
                    return x[:,:l,:]
                else:
                    return x


            #for x in loaded:
            #    print("X={}, Y={}".format(x[0].shape, x[1].shape))

            out_x = numpy.concatenate([padone(x) for x,y,l in loaded])
            
            
        out_y = numpy.concatenate([y for x,y,l in loaded])
        
        if self.multi_input:
            assert out_x0.shape[0] == out_y.shape[0]
            assert out_x0.shape[0] == len(loaded)
            assert out_x1.shape[0] == out_y.shape[0]
            assert out_x1.shape[0] == len(loaded)
        else:
            assert out_x.shape[0] == out_y.shape[0]
            assert out_x.shape[0] == len(loaded)
            
        if self.weights is not None:
            # compute weights
            out_w = numpy.array([self.weights[l] for x,y,l in loaded])
            self.cache[idx] = (out_x, out_y, out_w)
            return out_x, out_y, out_w
        else:
            self.cache[idx] = (out_x, out_y)
            return out_x, out_y
        
    # Others
    def int_ceildiv(self, a: int, b: int):
        return -((-a) // b)

    def time_normaliser(self, t):
        return numpy.log(numpy.maximum(0, t) * 1000 + 1)
    
    def domain_suffix(self, x):
        return '.'.join(x.split('.')[-2:])

    def mat_readone(self, f, sparse=True, verbose=False):
        loaded = scipy.io.loadmat(f)

        loaded_header = loaded["header"][:,:200]

        if loaded_header.shape[0] < 2:
            return None

        loaded_flow = loaded["flow"]
        loaded_label = loaded["label"][0]
        loaded_label = 1 if (self.domain_suffix(loaded_label) in self.domains) else 0

        if verbose:
            print("File={}".format(f))
            print("Shape: H={}, F={}".format(loaded_header.shape, loaded_flow.shape))
            print("Header={}, Flow={}, Label={}".format(loaded_header, loaded_flow, loaded_label))
            print("[0] H={}, F={}".format(loaded_header[0], loaded_flow))

        assert loaded_header.shape[0] == loaded_flow.shape[0]

        # Normalise
        loaded_header = loaded_header / 255
        loaded_flow = self.time_normaliser(loaded_flow)

        if verbose:
            print("Shapes: H={}, F={}".format(loaded_header.shape, loaded_flow.shape))

        
        if self.multi_input:
            out_x = (numpy.array([loaded_header]), numpy.array([loaded_flow]))
        else:
            out_x = numpy.concatenate([loaded_header, loaded_flow], axis=1)
            out_x = numpy.array([out_x])
            
        out_y = loaded_label

        out_y = numpy.array([out_y])
        
        return out_x, out_y, loaded_label