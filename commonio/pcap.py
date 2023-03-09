from scapy.all import *
from scapy.layers.ssl_tls import *
from scapy.error import Scapy_Exception

import numpy
import scipy.io
import os
import pickle
import random


def _get_zero_address(pkt):
    return '0.0.0.0' if IP in pkt else '::1'
def _get_ip_layer(pkt):
    return pkt.getlayer(IP) if IP in pkt \
        else pkt.getlayer(IPv6)
def _str_to_vec(s):
    return numpy.fromstring(s, dtype='uint8')
def _get_time(pkt):
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

def packet_to_vec(packet, truncate=200, mask=True, replace_servername=None):
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
    
def pcap_to_vec(pcap, truncate=200, seed=0):
    random.seed(seed)
    
    len_servername = random.randint(2, 30)
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
        
        packetTime = _get_time(packet)
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
        v_header = packet_to_vec(packet, truncate, mask=True, replace_servername=servername)
        
        result_header = numpy.append(result_header, [v_header], axis=0)
        
        
    assert np == result_header.shape[0]
    assert result_header.shape[0] == result_flow.shape[0]
    
    return result_header, result_flow

def pcap_to_mat(doc):
    """
    Produces a byte-vectorised packet for each packet in pcap
    """
    assert type(doc) == dict
    
    fpath = doc["fpath"]
    pcap = rdpcap(fpath)

    truncate = 1500
    
    error = None
    try:
        result_header, result_flow = pcap_to_vec(pcap, truncate=truncate, seed=fpath)
    except Exception as ex:
        result_header, result_flow = None,None
        error = ex
        
    result_label = doc["standalone_label"] if doc["standalone_label"] and doc["standalone_label"] != b"N/A" \
        else doc["group_label"]
    
    
    return fpath, { "error": error }, {"header": result_header, "flow": result_flow, "label": result_label }
