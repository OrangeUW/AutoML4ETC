#from kabab.flow import check_tls, size_seq, dir_seq, inter_arrival
from kabab.utils.general import get_label, read_inputs, get_pcaps, load_pcap
from kabab.utils import get_logger
from kabab.utils.gen import force_length
from kabab.config import FEATURE_SIZE
from kabab.utils.sprk import get_spark_session, read_csv, write_csv

from pyspark.sql import Row


import shutil, path

import numpy as N
import pandas as P

from scapy.all import *
from scapy.error import Scapy_Exception

from scipy.io import savemat

from commonio.pcap import *
from pathlib import Path

import unittest

#### Pcap

def execute_pcap_command(arg):
    """executes the pcap_function (func) from the to_flow_pcap_commands function
    """
    
    fpath, (pcap, pcap_func) = arg
    return fpath, pcap_func(pcap)

def _flow_info(pcap):
    """extracts flow information from pcap
    
    Args:
        pcap (pcap obj.)
        
    Returns: 
        (dict) : flow information dictionary
    
    """
    assert all([IP in pkt or IPv6 in pkt for pkt in pcap])
    
    L3 = IPv6 if IPv6 in pcap[0] else IP
    
    return {
        "num_packets": len(pcap),
        "flow_bytes": sum(((pkt.plen + 40) if IPv6 in pkt else pkt.len for pkt in pcap)),
        "flow_available_bytes": sum((len(pkt) for pkt in pcap)),
        "start_time": pcap[0][L3].time,
        "end_time": pcap[-1][L3].time,
        "src": pcap[0][L3].src,
        "dst": pcap[0][L3].dst,
        "sport": pcap[0].sport,
        "dport": pcap[0].dport
    }

def filter_tls(path_pcap):
    """filters tls pcaps
    
    Args:
        path_pcap (str, pcap obj.)
       
    Returns:
        (bool) : true if the pcap contains at least one TLS packet, else false
    
    """
    path, pcap = path_pcap
    return any([TLS in pkt for pkt in pcap])


#NEW FOR TLS FILTER
import io
from contextlib import redirect_stdout


def filter_tls2(path_pcaps):
    """ Same as the filter_tls function
    
        another implementation of filter_tls function for diagnosis purposes of Scapy
            
    """
    path, pcaps = path_pcaps
    G = io.StringIO()

    with redirect_stdout(G):
        for pkt in pcaps:
            pkt.show()
                
    s = G.getvalue()
    
    return "TLS" in s

def _is_ip_internal(ipaddr):
    return ipaddr.startswith("51.81.246.223")
def _is_direction_forward(doc):
    if doc["dport"] == 443:
        return True
    if doc["sport"] == 443:
        return False
    if _is_ip_internal(doc["src"]):
        return True
    if _is_ip_internal(doc["dst"]):
        return False
    raise Exception("Can't establish the direction of document: {}".format(doc))

def _unique_four_tuple(doc):
    if _is_direction_forward(doc):
        return (doc['src'], doc['dst'], doc['dport'])
    else:
        return (doc['dst'], doc['src'], doc['sport'])
    
def _separate_per_timeout(doc_list, idle_timeout):
    
    res = []
    cur_flow_group = []
    
    for flow in doc_list:
        if len(cur_flow_group) > 0 and flow["start_time"] - cur_flow_group[-1]["end_time"] > idle_timeout:
            res.append(cur_flow_group)
            cur_flow_group = []       
        cur_flow_group.append(flow)
    
    # add last batch (if any)
    if cur_flow_group:
        res.append(cur_flow_group)
        
    return res
def _first_sni(pcap, throw=True):
    """extracts SNI field from pcap
    
    Args:
        pcap (pcap obj.)
        throw (bool) : if True, raise Exception
    
    """
    
    for pkt in pcap:
        #NEW added and data
        if TLSExtServerNameIndication in pkt and len(pkt.getlayer(TLSExtServerNameIndication).server_names) > 0:
            return pkt.getlayer(TLSExtServerNameIndication).server_names[0].data
    
    if throw:    
        raise Exception("No SNI")
    
    return b"N/A"

def _flow_label(pcap):
    """ extracts flow label from pcap by their SNI field in ClientHello packet
    
    Args:
        pcap (pcap obj.)
    
    Returns:
        (dict) : SNI field value
    """
    return {
        "standalone_label": _first_sni(pcap, throw=False),
    }

def to_flow_pcap_commands(flow_args):
    """this function generates high-level flow informations and label
    
    Args:
        flow_args (str, pcap obj.) : tuple of pcapPath and pcap object
        
    Returns:
        [
        (pcapPath, (pcap obj, func),  : function object for extracting flow informations
        (pcapPath, (pcap obj, func),  : function object for extracting flow label
        ]
    """
    
    fpath, pcap = flow_args
    return [
        (fpath, (pcap, lambda pcap: _flow_info(pcap))),
        (fpath, (pcap, lambda pcap: _flow_label(pcap)))
    ]

def _extract_pkt_alpn(pkt, layer=None):

    if layer is None:
        if TLSServerHello in pkt:
            layer = TLSServerHello
        elif TLSClientHello in pkt:
            layer = TLSClientHello
    
    if layer not in pkt:
            return []

    protocols = []

    for extension in pkt.getlayer(layer).extensions:
        if hasattr(extension, "type") and hasattr(extension, "protocol_name_list") and extension.type == 16:
            for protocol in extension.protocol_name_list:
                protocols.append(protocol.data)

    return protocols

def _extract_protocols(pcap, layer=None):
    return set([prot for pkt in pcap for prot in _extract_pkt_alpn(pkt, layer)])

def _group_info(super_pcap):
    return {
        "group_cli_prots": _extract_protocols(super_pcap, layer=TLSClientHello),
        "group_srv_prots": _extract_protocols(super_pcap, layer=TLSServerHello),
        "group_label": _first_sni(super_pcap, throw=False),
    }

## Packet to vector

def _get_zero_address(pkt):
    return '0.0.0.0' if IP in pkt else '::1'
def _get_ip_layer(pkt):
    return pkt.getlayer(IP) if IP in pkt \
        else pkt.getlayer(IPv6)
def _str_to_vec(s: str):
    return numpy.fromstring(s, dtype='uint8')
def _get_time(pkt):
    """
    Warning: This function is not deterministic. Use pkt.time!
    """
    if IP in pkt:
        return pkt[IP].time
    else:
        assert IPv6 in pkt
        return pkt[IPv6].time
def _get_len(pkt):
    if IPv6 in pkt:
        return pkt.plen + 40
    else:
        return pkt.len

def _packet_to_vec(packet, truncate=200, mask=True, replace_servername=None):
    assert truncate != 200
    
    # Throw away all packets not IP or IPv6, not TCP and not UDP
    if not (IP in packet or IPv6 in packet):
        return None
    if not (UDP in packet or TCP in packet):
        return None
            
    assert IP in packet or IPv6 in packet
    
    netlayer = _get_ip_layer(packet).copy()
    # mask IP
    zero = _get_zero_address(packet)
    
    if mask:
        netlayer.src = zero
        netlayer.dst = zero
    
        #mask TCP/UDP port
        if UDP in packet or TCP in packet:
            netlayer.sport = 0
            netlayer.dport = 0
        
    # Add IP header
    header_length = len(netlayer) - len(netlayer.payload)
    mbytes = bytes(netlayer)[:header_length]
    
    # zero-pad UDP header so its of the same length as TCP
    if UDP in packet:
        mbytes += bytes(netlayer.getlayer(UDP))[:8]
        mbytes += bytes('\0' * 12, encoding='utf-8')
            
    assert len(mbytes) <= truncate
    
    if TLS in packet:
        tlslayer = packet[TLS]
        if TLSCiphertext in tlslayer:
            del tlslayer[TLSCiphertext]
        if TLSExtServerNameIndication in tlslayer and replace_servername:
            n_names = len(tlslayer[TLSExtServerNameIndication].server_names)
            replacement = [TLSServerName(replace_servername) for _ in range(n_names)]
            tlslayer[TLSExtServerNameIndication].server_names = replacement
          
        mbytes += bytes(tlslayer)[:len(tlslayer) - len(tlslayer.payload)]
    mbytes = mbytes[:truncate]
    
    row = numpy.zeros(shape=(truncate,), dtype='uint8')
    row[:len(mbytes)] = _str_to_vec(mbytes)
    return row
     
def _pcap_to_vec(pcap, truncate=500, seed=0):
    random.seed(seed)
    
    URL_LENGTH_MU = 20.53
    URL_LENGTH_SIGMA = 6.47
    
    len_servername = max(int(numpy.random.normal(URL_LENGTH_MU, URL_LENGTH_SIGMA)), 4)
    servername = TLSServerName(bytes([random.getrandbits(7) for _ in range(len_servername)]))
    
    result_header = numpy.empty(shape=(0, truncate), dtype='uint8')
    result_flow = numpy.empty(shape=(0, 2), dtype='float32')
    
    lastTime = 0
    np = 0
    
    addr_source = None
    for packet in pcap:
        # Throw away all packets not IP or IPv6, not TCP and not UDP
        if not (IP in packet or IPv6 in packet):
            continue
        if not (UDP in packet or TCP in packet):
            continue
            
        np += 1
        netlayer = _get_ip_layer(packet).copy()
        # mask IP
        zero = _get_zero_address(packet)
        
        if addr_source == None:
            addr_source = netlayer.src
        direction = netlayer.src == addr_source
        
        packetTime = packet.time #_get_time(packet)
        if lastTime == 0:
            irt = 0
        else:
            irt = packetTime - lastTime
        lastTime = packetTime
        
        size = _get_len(packet)
        if not direction:
            size = -size
        v_flow = numpy.array([irt, size])
        
        result_flow = numpy.append(result_flow, [v_flow], axis=0)
        
        
        # The step below destroys some vital information in the packet so it has to be done last.
        v_header = _packet_to_vec(packet, truncate, mask=True, replace_servername=servername)
        
        result_header = numpy.append(result_header, [v_header], axis=0)
        
    assert np == result_header.shape[0]
    assert result_header.shape[0] == result_flow.shape[0]
    
    return result_header, result_flow

#### Dictionaries


def merge_dicts(d1, d2):
    # Can write this as { **d1, **d2 }
    res = dict()
    res.update(d1)
    res.update(d2)
    return res

REMOVED_KEYS = [
    'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Label',
    #'Flow Byts/s',
    #'Flow Pkts/s', 
    #'Bwd PSH Flags', 
    #'Bwd URG Flags',
    #'ECE Flag Cnt',
    #'Fwd Byts/b Avg',
    #'Fwd Pkts/b Avg',
    #'Fwd Blk Rate Avg',
    #'Bwd Byts/b Avg',
    #'Subflow Bwd Pkts',
    #'Active Mean',
    #'Active Std',
    #'Active Max',
    #'Active Min',
    #'Idle Mean', 
    #'Idle Std',
    #'Idle Max',
    #'Idle Min',
]



class FlowPreprocessor:
    
    def __init__(self,
                 pcapPath,
                 outPath,
                 cachePath=None,
                 cfmPath=None,
                 sparkPartitions=80,
                 packetTruncate=1500,
                 idleTimeout=300):
        """Main class for initializing pcap preprocsessing
        
        Args:
            pcapPath (str) : Root path to all pcap files
            outPath: (str) : Path that contains all .mat files, one for each pcap
            cachePath: (str) : Path for symbolic links for cfm
            cfmPath (str) : Path of cfm from CIC flow meter
            packetTruncate (int) : TODO
            idleTimeout (int) : TODO
        """
        self.pcapPath = Path(pcapPath)
        self.outPath = Path(outPath)
        self.cachePath = Path(cachePath) if cachePath else None
        self.cfmPath = Path(cfmPath) if cfmPath else None
        
        self.sparkPartitions = sparkPartitions
        self.packetTruncate = packetTruncate
        self.idleTimeout = idleTimeout
        
    def path_to_mat(self, path):
        """
        Convert a .pcap path to the corresponding .mat path
        """
        return self.outPath / path.relative_to(self.pcapPath).with_suffix('.mat')
    def path_to_cache(self, path):
        """
        Convert a pcap to its shadow path for symlinks
        """
        p2 = path.relative_to(self.pcapPath)
        return self.cachePath / str(p2).replace('/', '_')
    
    def path_to_csv(self, path):
        """
        Convert a .pcap path to the corresponding .csv path
        """
        p2 = path.relative_to(self.pcapPath)
        p2 = self.cachePath / "out" / str(p2).replace('/', '_')
        return p2.with_name(p2.name + '_Flow.csv')
    
    def datapath_to_mat(self, datapath):
        return str(self.outPath / Path(datapath + '.mat'))
    def path_to_datapath(self, p):
        """returns a relative path to pcapPath
            
            Args:
                datapath (str) : path
            
            Returns:
                (str) : relative path to pcapPath
            
        """
        
        return str(Path(p).relative_to(self.pcapPath).with_suffix(''))
            
    def _process_pcap(self, doc):
        """
        Produces a byte-vectorised packet for each packet in pcap
        """
        assert type(doc) == dict

        fpath = doc["fpath"]
        pcap = rdpcap(fpath)

        error = None
        try:
            result_header, result_flow = _pcap_to_vec(pcap, truncate=self.packetTruncate, seed=fpath)
        except Exception as ex:
            result_header, result_flow = None,None
            error = ex

        result_label = doc["standalone_label"] if doc["standalone_label"] and doc["standalone_label"] != b"N/A" \
            else doc["group_label"]


        return fpath, { "error": error }, {"header": result_header, "flow": result_flow, "label": result_label }

    def _rdd_save_mat(self, arg):
        path, diag, result = arg
        path = Path(path)
        
        mpath = self.path_to_mat(path)
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        
        if diag['error'] is None:
            savemat(mpath, result)
        return (path, diag['error'])
    
    
    def read_directories(self, verbose=False, sample=False):
        """reads all .pcap files from pcapPath directory
        
        Args:
            verbose (bool) : print number of pcap files in pcapPath dir
            sample (int) : number of pcap files want to use | Do not use zero for this value!
        
        Returns: 
            pcapFiles (list) : absolute path to all pcap files in pcapPath
        
        """
        
        #Path.resolve -> converts relative path (including . and ..) to absolute path | ** -> all sub dirs 
        pcapFiles = list(self.pcapPath.resolve().glob('**/*.pcap'))
        if verbose:
            print("{} pcap files scanned".format(len(pcapFiles)))
        if sample:
            pcapFiles = pcapFiles[:sample]
        return pcapFiles

    def output_list(self, pcapFiles):
        """Outputs a list (plain text, delimited by \n) of data paths to all.txt in the output directory.
        
        The datapaths can be transformed back into mat files using self.datapath_to_mat.
        
        Args:
            pcapFiles (list) : list of Path objects 
        
        """
        li = [self.path_to_datapath(x) for x in pcapFiles]
        
        with open(self.outPath / 'all.txt', 'w') as f:
            for x in li:
                f.write(x + '\n')
                
    def output_packets(self, pathsRdd, verbose=False):
        """Main function for .pcap files processing
        
        //TO DO add more comment
        
        Args:
            pathsRdd (Apache Spark RDD obj) : the RDD of pcapFiles 
        
        """
        
        def merge_path_into_dict(path_dic):
            path, dic = path_dic
            dic["fpath"] = path
            dic["datapath"] = self.path_to_datapath(path)
            return dic
        
        #in each line I will demonstrate the look of the current output of RDD
        flow_stats_rdd = pathsRdd \
            .map(load_pcap) \
            .filter(filter_tls) \
            .flatMap(to_flow_pcap_commands) \
            .map(execute_pcap_command) \
            .reduceByKey(merge_dicts) \
            .map(merge_path_into_dict) \
            .repartition(self.sparkPartitions) \
            .cache()
        
        flow_groups_rdd = flow_stats_rdd \
            .groupBy(_unique_four_tuple) \
            .map(lambda x: sorted(x[1], key=lambda x: x["start_time"])) \
            .flatMap(lambda x:_separate_per_timeout(x, self.idleTimeout)) \
            .repartition(self.sparkPartitions)
        
        super_flows_rdd = flow_groups_rdd \
            .map(lambda flowgroup: (flowgroup, [pkt for f in flowgroup for pkt in load_pcap(f["fpath"])[1]]))
        
        flow_labels_rdd = super_flows_rdd \
            .map(lambda fg_sf: (fg_sf[0], _group_info(fg_sf[1]))) \
            .flatMap(lambda fg_label: [{**flow, **(fg_label[1])} for flow in fg_label[0]]) \
            .repartition(self.sparkPartitions)
        
        nTLS = flow_labels_rdd.count()
        if verbose:
            print("Extracting flow groups and saving .mat files ...")
            print("Total TLS count: {}".format(nTLS))
        
        
        errors = flow_labels_rdd \
            .map(self._process_pcap) \
            .map(self._rdd_save_mat) \
            .filter(lambda x:x[1] is not None) \
            .collect()
        nErrors = len(errors)
        #NEW 
        if nTLS != 0 and verbose:
            print("Conversion fail rate: {}/{}".format(nErrors, nTLS))
        elif verbose and nTLS == 0:
            print("Conversion fail rate: nErrors:{},nTLS{}".format(nErrors, nTLS))

        
        flow_groups_processed = flow_groups_rdd.collect()
        
        with open(self.outPath / 'flowgroups.pickle', "wb") as f:
            pickle.dump(flow_groups_processed, f)
            
        def extract_meta(doc):
            """
            Find which packet has client hello
            """
            assert type(doc) == dict
            
            fpath = doc["fpath"]
            datapath = self.path_to_datapath(fpath)
            pcap = rdpcap(fpath)
            
            handshake_positions = [i for i in range(len(pcap)) if TLSHandshake in pcap[i]]
            if handshake_positions != []:
                min_handshake = min(handshake_positions)
            else:
                min_handshake = -1
                
            result = {
                "iHandshake": min_handshake,
                "handshake_positions": handshake_positions,
                "group_cli_prots": _extract_protocols(pcap, layer=TLSClientHello),
                "group_srv_prots": _extract_protocols(pcap, layer=TLSServerHello),
                "group_label": _first_sni(pcap, throw=False),
            }
            return datapath, result
        
        
        if verbose:
            print(f"{len(flow_groups_processed)} groups extracted")
            print("Extracting metadata ...")
            
        meta = flow_labels_rdd \
            .map(extract_meta) \
            .collect()
        meta = dict(meta)
        
        with open(self.outPath / 'metadata.pickle', "wb") as f:
            pickle.dump(meta, f)
        
        return errors
        
    ### Flow meter stats
    
    @property
    def _cfm_command(self):
        return ['./' + self.cfmPath.name, str(self.cachePath), str(self.cachePath / 'out')]
    
    def _execute_cfm(self, pathsRdd, verbose=True):
        assert self.cfmPath, "CICFlowMeter path not provided"
        
        if verbose:
            print('Building symbolic links ...')
        
                  
        os.makedirs(self.cachePath, exist_ok=True)
        def linkbuilder(path):
            cachePath = self.path_to_cache(path)
            cachePath.symlink_to(path)
        
        pathsRdd.foreach(linkbuilder)
        
        if verbose:
            print('Executing cfm ...')
            
        subprocess.check_output(self._cfm_command, cwd=self.cfmPath.parent)
        
    def output_flows(self, pathsRdd, verbose=True):
        
        #pathsRdd = sc.parallelize(self.pcapFiles, self.sparkPartitions)
        self._execute_cfm(pathsRdd, verbose=verbose)
        
        if verbose:
            print('Reading cfm stats ...')
            
        def readOne(path):
            csvPath = self.path_to_csv(path)
            dataPath = self.path_to_datapath(path)
            try:
                tab = P.read_csv(csvPath).drop(REMOVED_KEYS, axis=1)
                tab = tab.astype(float).to_numpy()
                return dataPath, tab[0], tab.shape[0]
            except:
                return dataPath, None, -1
            
        
        pathreadRdd = pathsRdd.map(readOne).cache()
        
        nsubflows = pathreadRdd.map(lambda x:x[2]).collect()
        
        validList = pathreadRdd \
            .filter(lambda x:x[2] >= 1) \
            .map(lambda x:(x[0],x[1])) \
            .collect()
        #return validList
        validList,matrix = zip(*validList)
        matrix = N.stack(matrix)
        assert matrix.shape[0] == len(validList)
        
        
        with open(self.outPath / 'all_cfm.txt', 'w') as f:
            for x in validList:
                f.write(x + '\n')
                
        nn = len(validList)
        nd = pathsRdd.count()
        print(f"{nn}/{nd} = {nn/nd:.3f} paths have flow information.")
        
        ne = sum(x == -1 for x in nsubflows)
        if ne > 0:
            print(f"{ne}/{nd} = {ne/nd:.3f} paths flow information cannot be loaded. These are indicated with a -1 in the number of subflows.")
        
        mu = N.mean(matrix, axis=0)
        sigma = N.std(matrix, axis=0)
        usable_cols = (~N.isnan(sigma)) & (~N.isnan(mu)) & (sigma > 0)
        
        assert mu.shape == (matrix.shape[1],)
        assert sigma.shape == (matrix.shape[1],)
        assert usable_cols.shape == (matrix.shape[1],)
        
        #matrix = (matrix - mu) / sigma
        
        savemat(self.outPath / 'cfm.mat', {
            'flows': matrix,
            'mu': mu,
            'sigma': sigma,
            'usable_cols': usable_cols,
            'nsubflows': N.array(nsubflows),
        })
        
        return validList, matrix
        
    def clear_cache(self):
        shutil.rmtree(self.cachePath)
    def clear_output(self):
        shutil.rmtree(self.outPath)
        


                
class TestPreprocessing(unittest.TestCase):
    
    def test_flow_paths(self):
        preproc = FlowPreprocessor(sparkContext=None,
                                   pcapPath="/home/dataset/orange/pcaps",
                                   outPath="/home/dataset/orange/mat",
                                   cachePath="/home/dataset/orange/cache",
                                   cfmPath='/home/bin/CICFlowMeter/cfm')
        path1 = Path('/home/dataset/orange/pcaps/batch123/flow5.pcap')
        self.assertEqual(preproc.path_to_mat(path1), Path('/home/dataset/orange/mat/batch123/flow5.mat'))
        self.assertEqual(preproc.path_to_cache(path1), Path('/home/dataset/orange/cache/batch123_flow5.pcap'))
        self.assertEqual(preproc.path_to_csv(path1), Path('/home/dataset/orange/cache/csv/batch123_flow5.pcap_Flow.csv'))
        self.assertEqual(preproc._cfm_command, ['./cfm', '/home/dataset/orange/cache', '/home/dataset/orange/cache/out'])
        
if __name__ == '__main__':
    unittest.main()
