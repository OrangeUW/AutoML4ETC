from kabab.flow import check_tls, size_seq, dir_seq, inter_arrival
from kabab.utils.general import get_label, read_inputs, get_pcaps, load_pcap
from kabab.utils import get_logger
from kabab.utils.gen import force_length
from kabab.config import PARTITIONS

from pyspark.sql import Row

from kabab.config import FEATURE_SIZE
from kabab.utils.sprk import get_spark_session, read_csv, write_csv

import shutil
from path import Path

from scapy.all import *
from scapy.layers.ssl_tls import *
from scapy.error import Scapy_Exception

import numpy
import scipy.io
import os

def execute_pcap_command(arg):
    fpath, (pcap, pcap_func) = arg
    return fpath, pcap_func(pcap)


def _flow_info(pcap):

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

def to_flow_pcap_commands(flow_args):
    fpath, pcap = flow_args
    return [
        (fpath, (pcap, lambda pcap: _flow_info(pcap))),
    ]

def filter_tls(path_pcap):
    path, pcap = path_pcap
    return any([TLS in pkt for pkt in pcap])


def merge_path_into_dict(path_dic):
    path, dic = path_dic
    dic["fpath"] = path
    return dic

IDLE_TIMEOUT = 300

# NOTE this will change from dataset to dataset
def __is_ip_internal(ipaddr):
    return ipaddr.startswith("10.") or ipaddr.startswith("192.168.") or ipaddr

def __is_direction_forward(doc):
    if doc["dport"] == 443:
        return True
    if doc["sport"] == 443:
        return False
    if __is_ip_internal(doc["src"]):
        return True
    if __is_ip_internal(doc["dst"]):
        return False
    raise Exception("Can't establish the direction of document: {}".format(doc))

def __unique_four_tuple(doc):
    if __is_direction_forward(doc):
        return (doc['src'], doc['dst'], doc['dport'])
    else:
        return (doc['dst'], doc['src'], doc['sport'])
    
def __separate_per_timeout(doc_list, idle_timeout=IDLE_TIMEOUT):
    
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

def merge_dicts(x, y):
    return {**x, **y}


def superflow_rdd(paths_rdd):
    # rdd is spark rdd consisting of pcap file paths
    return paths_rdd \
        .map(load_pcap) \
        .filter(filter_tls) \
        .flatMap(to_flow_pcap_commands) \
        .map(execute_pcap_command) \
        .reduceByKey(merge_dicts) \
        .map(merge_path_into_dict) \
        .groupBy(__unique_four_tuple) \
        .map(lambda x: sorted(x[1], key=lambda x: x["start_time"])) \
        .flatMap(__separate_per_timeout)