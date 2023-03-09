import scipy.io
import scipy.signal
import math
import numpy as N
import tensorflow.keras.utils
import pickle
from expiringdict import ExpiringDict


class DataGenSuperflow(tensorflow.keras.utils.Sequence):
    
    def __init__(self, files, label_func, weight_func=None,
                 idxfilter=lambda x: True, \
                 batchsize=16, weights=None, \
                 header_truncate=256, \
                 output_flow = False, \
                 path_transform_func = lambda x: x, \
                 n_packets=20, n_flows=1000, header_selector=None, \
                 flow_separate_features=False, stft=None, \
                 superflow_truncate=None, \
                 compress_y=True, \
                 auxiliary=None, auxiliary_meta=None, \
                 add_start_time=True, \
                 fallback_label_func=None, \
                 mask=-1):
        """
        files: [[str]], contains a list of list of .mat files
        idxfilter: Filter the files by their index to generate training/test datasets.
        batchsize: Number of files (samples) for each batch
        weights: Either None, or an array of size NUM_TYPES for weighing the classes
        sparse: If false, categorical variables are encoded one-hot.
        header_truncate: If a positive number is provided, trim header to given size
        fallback_label_func: If label does not exist, the function will get called with datapath
        
        header_selector: If a function is provided, only one header packet will be selected based on header_selector
        
        Default settings are consistent with Rezaei et al.
        """
        self.idxfilter = idxfilter
        self.header_selector = header_selector
        self.files = files
        self.files = [f for idx,f in enumerate(self.files)if self.idxfilter(idx)]
        #if self.header_selector:
        #    def hasHeader(li):
        #        hli = [self.header_selector(f['fpath_mat']) for f in li]
        #        return all([isinstance(h, list) or h >= 0 for h in hli])
        #    self.files = [li for li in self.files if hasHeader(li)]
        self.batchsize = batchsize
        self.cache = ExpiringDict(max_len=8, max_age_seconds=3600)
        self.weights = weights
        self.header_truncate = header_truncate
        self.n_packets = n_packets
        self.n_flows = n_flows
        self.superflow_truncate = superflow_truncate
        self.flow_separate_features = flow_separate_features
        self.stft = stft
        self.compress_y = compress_y
        self.add_start_time = add_start_time
        self.fallback_label_func = fallback_label_func
        
        self.label_func = label_func
        self.weight_func = weight_func
        self.output_flow = output_flow
        self.path_transform_func = path_transform_func
        self.mask = mask
        
        
        self.auxiliary = auxiliary
        self.auxiliary_meta = auxiliary_meta
        
        self.epsilon = 0.001
        
    def __len__(self):
        return self.int_ceildiv(len(self.files), self.batchsize)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache.get(idx)
        
        assert hasattr(self, 'header_truncate')
        
        start_idx = idx * self.batchsize
        loaded = [x for x in self.files[start_idx:start_idx+self.batchsize]]
        loaded = [self.flow_readone(x) for x in loaded]
        
        starttimes, out_x0, out_x1, out_y, out_w = zip(*loaded)
        
        starttimes = self.justified_merge2(starttimes)
        out_x0 = self.justified_merge(out_x0)
        out_x1 = self.justified_merge(out_x1)
        
        if self.superflow_truncate is not None:
            starttimes = starttimes[:,:self.superflow_truncate]
            out_x0 = out_x0[:,:self.superflow_truncate,:,:]
            out_x1 = out_x1[:,:self.superflow_truncate,:,:]

        out_x = [out_x0]
        if self.output_flow:
            out_x.append(out_x1)
        if self.add_start_time:
            out_x.append(starttimes)
        if self.auxiliary:
            out_aux = [self.aux_readone(x) for x in self.files[start_idx:start_idx+self.batchsize]]
            out_aux = self.justified_merge3(out_aux)
            if self.superflow_truncate is not None:
                out_aux = out_aux[:,:self.superflow_truncate]
            out_x.append(out_aux)

        if self.compress_y:
            out_y = N.stack(out_y)
        else:
            out_y = self.justified_merge2(out_y, mask=-2)
        
        if self.weight_func:
            out_w = N.concatenate(out_w)
            self.cache[idx] = (out_x, out_y, out_w)
            return out_x, out_y, out_w
        else:
            self.cache[idx] = (out_x, out_y)
            return out_x, out_y
    
    def justified_merge(self, li, max_axis3=None):
        max_axis1 = max([x.shape[0] for x in li])
        max_axis2 = max([x.shape[1] for x in li])
        if max_axis3 is None:
            max_axis3 = max([x.shape[2] for x in li])
        
        def padone(x):
            x = x[N.newaxis,:]
            x = N.pad(x, ((0,0),(0, max_axis1 - x.shape[1]), (0, max_axis2 - x.shape[2]), (0, max_axis3 - x.shape[3])), \
                             'constant', constant_values=self.mask)
            if x.shape[-1] < max_axis3:
                x = N.pad(x, ((0,0),(0,0),(0,0),(0,max_axis3 - x.shape[3])), 'constant', constant_values=self.mask)
            else:
                x = x[...,:max_axis3]
            return x
        
        result = N.concatenate([padone(x) for x in li])
        assert result.shape[0] == len(li)
        return result
    
    def justified_merge3(self, li):
        max_axis1 = max([x.shape[0] for x in li])
        
        def padone(x):
            return N.pad(x, ((0, max_axis1 - x.shape[0]), (0,0)), \
                             'constant', constant_values=self.mask)
        
        result = N.stack([padone(x) for x in li])
        assert result.shape[0] == len(li)
        return result
    
    def justified_merge2(self, li, mask=None):
        if mask is None:
            mask = self.mask
        max_axis1 = max([x.shape[0] for x in li])
        
        def padone(x):
            x = N.array(x)[N.newaxis,:]
            return N.pad(x, ((0,0),(0, max_axis1 - x.shape[1])), \
                             'constant', constant_values=mask)
        
        result = N.concatenate([padone(x) for x in li])
        assert result.shape[0] == len(li)
        return result
        
    # Others
    def int_ceildiv(self, a: int, b: int):
        return -((-a) // b)

    def time_normaliser(self, t):
        #return N.log(N.maximum(0, t) * 1000 + 1)
        return N.sign(t) * N.log(N.abs(t) * 1000 + 1)
    def size_normaliser(self, s):
        # 128 being mean packet size
        return s / 128
    
    def domain_suffix(self, x):
        return '.'.join(x.split('.')[-2:])
    
    def flow_readone(self, li):
        starttimes = [x['start_time'] for x in li]
        max_starttime = max(starttimes)
        starttimes = [self.time_normaliser(max_starttime - x) for x in starttimes]
        starttimes = N.array(starttimes)
        
        results = [self.mat_readone(x['datapath']) for x in li]
        out_xs_header, out_xs_flow, out_ys, out_ws = zip(*results)
        
        def justified_merge_3d(xs):
            truncate = max([x.shape[1] for x in xs])
            
            result = [N.pad(x, ((0,0),(0,truncate-x.shape[1]),(0,0)), 'constant', constant_values=self.mask)
                      for x in xs]
            result = N.concatenate(result, axis=0)
            return result
        
        out_xs_header = justified_merge_3d(out_xs_header)
        out_xs_flow = justified_merge_3d(out_xs_flow)
        if self.compress_y:
            idxs, = N.where(N.array(out_ys) >= 0)
            assert len(idxs) > 0
            out_y = out_ys[idxs[0]]
        else:
            out_y = N.array(out_ys)
        out_w = N.array(out_ws[0])
        
        assert len(out_xs_header.shape) == 3
        assert len(out_xs_flow.shape) == 3
        #assert len(out_y.shape) == 1
        assert len(out_w.shape) == 1
        
        return starttimes, out_xs_header, out_xs_flow, out_y, out_w
     
    def aux_readone(self, li):
        aux = [self.auxiliary.get(x['datapath'], N.zeros((77,))) for x in li]
        try:
            aux = N.stack(aux)
        except:
            for x in aux:
                print(x.shape)
            raise
        if self.auxiliary_meta:
            aux = N.nan_to_num(aux, copy=True,
                               nan=self.auxiliary_meta['v_nan'],
                               posinf=self.auxiliary_meta['v_posinf'],
                               neginf=self.auxiliary_meta['v_neginf'])
            if 'mask' in self.auxiliary_meta:
                idx_mask = self.auxiliary_meta['mask']
                aux = aux[...,idx_mask]
                if 'mean' in self.auxiliary_meta:
                    mu = self.auxiliary_meta['mean'][idx_mask]
                    std = self.auxiliary_meta['std'][idx_mask]
                    aux = (aux - mu) / std
            else:
                if 'mean' in self.auxiliary_meta:
                    mu = self.auxiliary_meta['mean']
                    std = self.auxiliary_meta['std']
                    aux = (aux - mu) / std
        assert aux.shape[0] == len(li)
        return aux

    def mat_readone(self, f, verbose=False):
        loaded = scipy.io.loadmat(self.path_transform_func(f))
        loaded_header = loaded["header"]
        if self.header_truncate:
            loaded_header = loaded_header[:,:self.header_truncate]
            
        if self.header_selector:
            offset = self.header_selector(f)
            if not isinstance(offset, list):
                assert offset >= 0
                offset = list(range(offset,offset+self.n_packets))
            else:
                offset = offset[:self.n_packets]
            loaded_header = loaded_header[offset,:]
            
        # Align to the number of packets given
        if self.n_packets:
            diff = loaded_header.shape[0] - self.n_packets
            if diff > 0:
                loaded_header = loaded_header[:self.n_packets,:]
            elif diff < 0:
                loaded_header = N.pad(loaded_header, ((0,-diff),(0,0)), \
                                      'constant', constant_values=self.mask)
                                                                
                    


        #print('flow: ',loaded['flow'].shape)
        loaded_flow = loaded["flow"][:self.n_flows,:]
        if loaded_flow.shape[0] < self.n_flows:
            loaded_flow = N.pad(loaded_flow, ((0,self.n_flows-loaded_flow.shape[0]),(0,0)),
                                'constant', constant_values=self.mask)
        
        loaded_label = loaded["label"][0]

        if loaded_label == -1 and self.fallback_label_func:
            loaded_label = self.fallback_label_func(f)
        
        if verbose:
            print("File={}".format(f))
            print("Shape: H={}, F={}".format(loaded_header.shape, loaded_flow.shape))
            print("Header={}, Flow={}, Label={}".format(loaded_header, loaded_flow, loaded_label))
            print("[0] H={}, F={}".format(loaded_header[0], loaded_flow))

        #assert loaded_header.shape[0] == loaded_flow.shape[0]

        # Normalise
        loaded_header = loaded_header.astype(float) / 255
        loaded_flow[:,0] = self.time_normaliser(loaded_flow[:,0])
        loaded_flow[:,1] = self.size_normaliser(loaded_flow[:,1])
        if self.flow_separate_features:
            loaded_flow = N.pad(loaded_flow, ((0,0),(0,1)), 'constant', constant_values=0)
            loaded_flow[:,2] = N.sign(loaded_flow[:,1])
            loaded_flow[:,1] = N.abs(loaded_flow[:,1])

        if self.stft is not None:
            #assert self.flow_use_dft == 'none'
            n_axis = loaded_flow.shape[1]
            
            def stft_load_one(i):
                x = loaded_flow[:,i]
                if self.n_flows > x.shape[0]:
                    x = N.pad(x, ((0,self.n_flows - x.shape[0]),), 'constant', constant_values=0)
                elif self.n_flows < x.shape[0]:
                    x = x[:self.n_flows]
                    
                #print(x.shape)
                assert x.shape[0] == self.n_flows
                assert x.ndim == 1
                
                assert self.stft['nperseg'] > self.stft['noverlap']
                _, _, S = scipy.signal.stft(x, 1, nperseg=self.stft['nperseg'], noverlap=self.stft['noverlap'])
                if self.stft['mode'] == 'abs':
                    return self.maxmin_normaliser(N.abs(S))
                elif self.stft['mode'] == 'cartesian':
                    p_real = self.maxmin_normaliser(N.real(S))
                    p_imag = self.maxmin_normaliser(N.imag(S))
                    return N.concatenate([p_real, p_imag], axis=0)
                elif self.stft['mode'] == 'rainbow' or self.stft['mode'] == 'polar':
                    mag = N.abs(S)
                    phase = N.angle(S)
                    
                    mag_max = N.amax(mag)
                    if mag_max > self.epsilon:
                        mag /= mag_max
                    
                    if self.stft['mode'] == 'rainbow':
                        phase = N.diff(phase, axis=-1, append=0)
                        phase /= 2 * math.pi
                    else:
                        phase /= math.pi
                    
                    return N.concatenate([mag, phase], axis=0)
                else:
                    assert False
            loaded_flow = [stft_load_one(i) for i in range(n_axis)]
            loaded_flow = N.concatenate(loaded_flow, axis=0)
            loaded_flow = N.swapaxes(loaded_flow,0,1)
            assert loaded_flow.ndim == 2
            
            
        if verbose:
            print("Shapes: H={}, F={}".format(loaded_header.shape, loaded_flow.shape))

        
        out_y = self.label_func(loaded_label)
            
            
        if self.weight_func:
            out_w = self.weight_func(out_y)
        else:
            out_w = 1

        out_w = N.array([out_w])
        
        return loaded_header[N.newaxis,:], loaded_flow[N.newaxis,:], out_y, out_w
