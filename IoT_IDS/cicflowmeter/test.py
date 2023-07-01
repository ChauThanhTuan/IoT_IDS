from scapy.all import *

packet = rdpcap("abc.pcapng")
print(packet.time)