import scipy.io
import scipy.signal
import math
import numpy as N
import tensorflow.keras.utils
import pickle
from expiringdict import ExpiringDict


class DataGenSeparated(tensorflow.keras.utils.Sequence):
    """
    Generator class for feeding large data-set of giant PCAPs parsed
    and pre-processed into multiple MAT files to the model
    
    This class is similar to DataGenOrange but uses a much higher truncation value for the flow sequence.
    """

    def __init__(self, files, label_func, weight_func=None,
                 idxfilter=lambda x: True,
                 batchsize=16,
                 n_packets=20, 
                 header_truncate=200, header_only=False, header_selector=None, header_mask=None,
                 path_transform_func=lambda x: x[x.index('/dataset/') + 1:],
                 aux_path_transform_func=lambda x: '/home/orange/soap-mctavish/' + x[x.index('/dataset/') + 1:],
                 n_flows=1000, flow_only=False, flow_separate_features=False, flow_use_dft='none', stft=None,
                 auxiliary=None, auxiliary_meta=None,
                 mask=0):
        """
        files: [str], contains a list of .mat files
        idxfilter: Filter the files by their index to generate training/test datasets.
        label_func: Integer label given a class
        weight_func: Generate a sample weight given a class (deprecated, please use keras weights)
        
        batchsize: Number of files (samples) for each batch
        
        -- Header --
        n_packets: Number of packets on the header side
        header_truncate: If a positive number is provided, trim header to given size (warning: check your preprocessor before setting this to something like none. If the preprocessor already "trims" all headers to a certain number of bytes this will have no effect)
        header_selector: If a function is provided, only one header packet will be selected based on header_selector
        header_only: If true, only the header input is provided
        
        -- Flow --
        n_flows: Number of packets on the flow side
        flow_only: If true, only the flow input is provided
        flow_separate_features: Decouple the direction & size information. Before decoupling, direction is stored as a sign on the size.
        flow_use_dft: Uses DFT on the flow side. One of {'none', 'abs', 'complex'}
        stft: Uses STFT on the flow side. Cannot be applied at the same time as dft. This must be a dict such that
            'mode': One of
                'abs' (modulus),
                'cartesian' (real, imag),
                'rainbow' (modulus, phase differential) which is from audio processing and looks smoother than polar,
                'polar' (modulus, phase)
            'nperseg': Same as nperseg in scipy.signal.stft
            'noverlap': Same as noverlap in scipy.signal.stft
        
        -- Auxiliary --
        auxiliary: If a dictionary of file -> 1D numpy array is fed, give the model an extra input.
        auxiliary_meta: Special information about standardisation, including:
            'v_nan': Value which should used to replace nan's,
            'v_posinf': ...,
            'v_neginf': ...,
            'mean', 'std': If exists, standardise the input using this mean. Note that mean and std must be either both provided or not provided,
            'mask': If exists, select the columns corresponding to these indices only. This is used to eliminate variables which convey no information or too much information.
        mask: Number used to pad inputs of different lengths
        """
        self.idxfilter = idxfilter
        self.header_selector = header_selector
        self.path_transform_func = path_transform_func
        self.aux_path_transform_func = aux_path_transform_func
        self.files = files
        self.files = [f for idx, f in enumerate(self.files) if self.idxfilter(idx)]
        if self.header_selector:
            def filt(f):
                h_index = self.header_selector(f)
                return type(h_index) == list or h_index >= 0

            self.files = [f for f in self.files if filt(f)]
        self.batchsize = batchsize
        self.cache = ExpiringDict(max_len=8, max_age_seconds=3600)

        self.n_packets = n_packets
        self.header_truncate = header_truncate
        self.header_mask = header_mask

        if self.header_mask is not None:
            self.header_mask = self.header_mask[N.newaxis, :]
            # self.header_mask = N.repeat(self.header_mask[N.newaxis,:], self.n_packets, axis=0)

        self.n_flows = n_flows
        self.flow_separate_features = flow_separate_features
        self.flow_use_dft = flow_use_dft
        self.stft = stft

        self.label_func = label_func
        self.weight_func = weight_func
        self.header_only = header_only
        self.flow_only = flow_only
        self.mask = mask
        self.epsilon = 0.001

        self.auxiliary = auxiliary
        self.auxiliary_meta = auxiliary_meta

    def __len__(self):
        return self.int_ceildiv(len(self.files), self.batchsize)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache.get(idx)

        assert hasattr(self, 'header_truncate')

        start_idx = idx * self.batchsize
        loaded = [x for x in self.files[start_idx:start_idx + self.batchsize]]
        loaded = [self.mat_readone(x) for x in loaded]
        loaded = [x for x in loaded if x]

        assert len(loaded) <= self.batchsize

        out_x0 = self.justified_merge([x[0] for x, y, w in loaded], self.mask, self.n_packets)
        try:
            out_x0_t = out_x0.reshape((len(loaded), self.n_packets, 5, self.header_truncate // 5))
            out_x0_t = N.swapaxes(out_x0_t, 1, 3)
        except:
            raise ValueError("reshaping to 3 chalnelles not working")
        
        out_x0 = out_x0_t
        #out_x0 = N.expand_dims(out_x0, axis=3)

        if self.stft is None:
            out_x1_align = self.n_flows
        else:
            out_x1_align = loaded[0][0][1].shape[1]
        out_x1 = self.justified_merge([x[1] for x, y, w in loaded], self.mask, out_x1_align)
        
        if self.flow_use_dft != 'none':
            out_x1 = N.fft.rfft(out_x1, axis=1)

            if self.flow_use_dft == 'abs':
                out_x1 = N.abs(out_x1)
            elif self.flow_use_dft == 'complex':
                out_x1 = N.concatenate([N.real(out_x1), N.imag(out_x1)], axis=-1)

            out_x1 = out_x1[:, 1:, :] / out_x1.shape[1]

        if self.auxiliary:
            out_x2 = self.justified_merge([x[2] for x, y, w in loaded], self.mask, None)

            if self.flow_only:
                out_x = [out_x1, out_x2]
            elif self.header_only:
                out_x = [out_x0, out_x2]
            else:
                out_x = [out_x0, out_x1, out_x2]
        else:
            if self.flow_only:
                out_x = out_x1
            elif self.header_only:
                out_x = out_x0
            else:
                out_x = [out_x0, out_x1]

        out_y = N.concatenate([y for x, y, w in loaded])
        out_w = N.concatenate([w for x, y, w in loaded])

        if self.weight_func:
            self.cache[idx] = (out_x, out_y, out_w)
            return out_x, out_y, out_w
        else:
            self.cache[idx] = (out_x, out_y)
            
            return out_x, out_y

    def justified_merge(self, li, mask, truncate=None):

        if not truncate:
            truncate = max([x.shape[1] for x in li])
            # print("truncation: {}, shape: {}, {}".format(truncate, li[0].shape, li[0].shape[1]))
            istruncated = False
        else:
            istruncated = True

        def padone(x):
            if truncate > x.shape[1]:
                pad = truncate - x.shape[1]
                return N.pad(x, ((0, 0), (0, pad), (0, 0)), 'constant', constant_values=mask)
            elif truncate < x.shape[1]:
                assert istruncated
                return x[:, :truncate, :]
            else:
                return x

        result = N.concatenate([padone(x) for x in li])
        assert result.shape[1] == truncate
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

    def maxmin_normaliser(self, x):
        return N.interp(x, (x.min(), x.max()), (0, 1))

    def domain_suffix(self, x):
        return '.'.join(x.split('.')[-2:])

    def mat_readone(self, f, verbose=False):

        loaded = scipy.io.loadmat(self.path_transform_func(f))

        if self.flow_only:
            loaded_header = N.array([[0]])
        elif self.header_truncate:
            loaded_header = loaded["header"][:, :self.header_truncate]
        else:
            loaded_header = loaded["header"]

        if self.header_mask is not None and self.header_selector:
            loaded_header = loaded_header[:self.n_packets]
            diff = self.n_packets - loaded_header.shape[0]
            if diff > 0:
                loaded_header = N.pad(loaded_header, ((0, diff), (0, 0)), 'constant', constant_values=self.mask)

            offset = self.header_selector(f)
            if type(offset) != list:
                offset = [offset]

            this_mask = [N.ones((1, self.header_truncate)) if (i in offset) else self.header_mask \
                         for i in range(self.n_packets)]
            this_mask = N.concatenate(this_mask)
            loaded_header *= this_mask.astype(loaded_header.dtype)
            pass
        elif self.header_mask is not None:
            loaded_header = loaded_header[:self.n_packets]
            diff = self.n_packets - loaded_header.shape[0]
            if diff > 0:
                loaded_header = N.pad(loaded_header, ((0, diff), (0, 0)), 'constant', constant_values=self.mask)

            loaded_header *= N.repeat(self.header_mask[N.newaxis, :], self.n_packets, axis=0).astype(
                loaded_header.dtype)
            pass
        elif self.header_selector:
            offset = self.header_selector(f)
            if type(offset) != list:
                assert offset >= 0
                offset = list(range(offset, offset + self.n_packets))
            else:
                offset = offset[:self.n_packets]
            loaded_header = loaded_header[offset, :]
            sizediff = self.n_packets - loaded_header.shape[0]
            assert sizediff >= 0
            if sizediff > 0:
                loaded_header = N.pad(loaded_header, ((0, sizediff), (0, 0)), 'constant', constant_values=self.mask)

        loaded_flow = loaded["flow"]
        loaded_label = loaded["label"][0]

        if verbose:
            print("File={}".format(f))
            print("Shape: H={}, F={}".format(loaded_header.shape, loaded_flow.shape))
            print("Header={}, Flow={}, Label={}".format(loaded_header, loaded_flow, loaded_label))
            print("[0] H={}, F={}".format(loaded_header[0], loaded_flow))

        # assert loaded_header.shape[0] == loaded_flow.shape[0]

        # Normalise
        loaded_header = loaded_header.astype(float) / 255
        loaded_flow[:, 0] = self.time_normaliser(loaded_flow[:, 0])
        loaded_flow[:, 1] = self.size_normaliser(loaded_flow[:, 1])

        if self.flow_separate_features:
            loaded_flow = N.pad(loaded_flow, ((0, 0), (0, 1)), 'constant', constant_values=0)
            loaded_flow[:, 2] = N.sign(loaded_flow[:, 1])
            loaded_flow[:, 1] = N.abs(loaded_flow[:, 1])

        if self.stft is not None:
            assert self.flow_use_dft == 'none'
            n_axis = loaded_flow.shape[1]

            def stft_load_one(i):
                x = loaded_flow[:, i]
                if self.n_flows > x.shape[0]:
                    x = N.pad(x, ((0, self.n_flows - x.shape[0]),), 'constant', constant_values=0)
                elif self.n_flows < x.shape[0]:
                    x = x[:self.n_flows]

                # print(x.shape)
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
            loaded_flow = N.swapaxes(loaded_flow, 0, 1)
            assert loaded_flow.ndim == 2

        if verbose:
            print("Shapes: H={}, F={}".format(loaded_header.shape, loaded_flow.shape))

        if self.auxiliary:
            aux = self.auxiliary[self.aux_path_transform_func(f)]
            if self.auxiliary_meta:
                aux = N.nan_to_num(aux, copy=True,
                                   nan=self.auxiliary_meta['v_nan'],
                                   posinf=self.auxiliary_meta['v_posinf'],
                                   neginf=self.auxiliary_meta['v_neginf'])
                if 'mask' in self.auxiliary_meta:
                    idx_mask = self.auxiliary_meta['mask']
                    aux = aux[idx_mask]
                    if 'mean' in self.auxiliary_meta:
                        mu = self.auxiliary_meta['mean'][idx_mask]
                        std = self.auxiliary_meta['std'][idx_mask]
                        aux = (aux - mu) / std
                else:
                    if 'mean' in self.auxiliary_meta:
                        mu = self.auxiliary_meta['mean']
                        std = self.auxiliary_meta['std']
                        aux = (aux - mu) / std

            out_x = (loaded_header[N.newaxis, :], loaded_flow[N.newaxis, :], aux[N.newaxis, :])
        else:
            out_x = (loaded_header[N.newaxis, :], loaded_flow[N.newaxis, :])

        out_y = self.label_func(loaded_label)

        if self.weight_func:
            out_w = self.weight_func(out_y)
        else:
            out_w = 1

        out_y = N.array([out_y])
        out_w = N.array([out_w])
        
        
        return out_x, out_y, out_w
