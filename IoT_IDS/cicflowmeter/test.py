import csv
from collections import defaultdict

from scapy.sessions import DefaultSession

from features.context.packet_direction import PacketDirection
from features.context.packet_flow_key import get_packet_flow_key
from flow import Flow


import netStat as ns
import numpy as np
import torch

from model import predict

EXPIRED_UPDATE = 40
MACHINE_LEARNING_API = "http://localhost:8000/predict"
# GARBAGE_COLLECT_PACKETS = 100
GARBAGE_COLLECT_PACKETS = 1

garbage = []
first_time = 0

class FlowSession(DefaultSession):
    """Creates a list of network flows."""

    def __init__(self, *args, **kwargs):
        self.flows = {}
        self.csv_line = 0

        ### Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)

        if self.output_mode == "flow":
            output = open(self.output_file, "w")
            self.csv_writer = csv.writer(output)

        self.packets_count = 0

        self.clumped_flows_per_label = defaultdict(list)

        super(FlowSession, self).__init__(*args, **kwargs)

    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        self.garbage_collect(None)
        return super(FlowSession, self).toPacketList()

    def on_packet_received(self, packet):
        global first_time, garbage
        current_time = packet.time

        if garbage == []:
            first_time = current_time
            garbage.append(packet)
        elif current_time - first_time > 10:
            first_time = current_time
            garbage = [packet]
        elif len(garbage)  == 32:
            for item in garbage:
                IPtype = np.nan
                timestamp = item.time
                framelen = len(item)
                if item.haslayer("IP"):  # IPv4
                    srcIP = item["IP"].src
                    dstIP = item["IP"].dst
                    IPtype = 0
                elif item.haslayer("IPv6"):  # ipv6
                    srcIP = item["IPv6"].src
                    dstIP = item["IPv6"].dst
                    IPtype = 1
                else:
                    srcIP = ''
                    dstIP = ''

                if item.haslayer("TCP"):
                    srcproto = str(item["TCP"].sport)
                    dstproto = str(item["TCP"].dport)
                elif item.haslayer("UDP"):
                    srcproto = str(item["UDP"].sport)
                    dstproto = str(item["UDP"].dport)
                else:
                    srcproto = ''
                    dstproto = ''

                srcMAC = item.src
                dstMAC = item.dst
                if srcproto == '':  # it's a L2/L1 level protocol
                    if item.haslayer("ARP"):  # is ARP
                        srcproto = 'arp'
                        dstproto = 'arp'
                        srcIP = item["ARP"].psrc  # src IP (ARP)
                        dstIP = item["ARP"].pdst  # dst IP (ARP)
                        IPtype = 0
                    elif item.haslayer("ICMP"):  # is ICMP
                        srcproto = 'icmp'
                        dstproto = 'icmp'
                        IPtype = 0
                    elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                        srcIP = item.src  # src MAC
                        dstIP = item.dst  # dst MAC


                ### Extract Features
                try:
                    data = self.nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,
                                                        int(framelen),
                                                        float(timestamp))
                    predict(data)
                    # inputs = DataPreprocessing(features)
                    # encoder = LabelEncoder()
                    # encoder.fit_transform(['benign', 'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_udp', 'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain'])
                    # # Take the random record from the test data
                    # result = []
                    # inputs_reshape = inputs.reshape(1, 115, 1)
                    # pred = model(inputs_reshape)
                    # pred_label = int(pred.argmax().data.cpu().numpy())
                    # pred_threat = encoder.inverse_transform([pred_label])[0]
                    # result.append(pred_threat)
                    # print("Predicted threat type: ", pred_threat)

                except Exception as e:
                    print(e)
                    return []

            first_time = 0
            garbage = []
        else:
            garbage.append(packet)
    


    def get_flows(self) -> list:
        return self.flows.values()

    def garbage_collect(self, latest_time) -> None:
        # TODO: Garbage Collection / Feature Extraction should have a separate thread
        if not self.url_model:
            print("Garbage Collection Began. Flows = {}".format(len(self.flows)))
        keys = list(self.flows.keys())
        for k in keys:
            flow = self.flows.get(k)

            if (
                latest_time is None
                or latest_time - flow.latest_timestamp > EXPIRED_UPDATE
                or flow.duration > 90
            ):
                data = flow.get_data()

                if self.csv_line == 0:
                    self.csv_writer.writerow(data.keys())

                self.csv_writer.writerow(data.values())
                self.csv_line += 1

                del self.flows[k]
        if not self.url_model:
            print("Garbage Collection Finished. Flows = {}".format(len(self.flows)))


def generate_session_class(output_mode, output_file, url_model):
    return type(
        "NewFlowSession",
        (FlowSession,),
        {
            "output_mode": output_mode,
            "output_file": output_file,
            "url_model": url_model,
        },
    )
