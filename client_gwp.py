import cv2
import numpy as np
import socket
import sys
import pickle
import struct ### new code
HOST = '128.32.112.46'
PORT = 8089


cap=cv2.VideoCapture(0)
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect((HOST, PORT))
while True:
    ret,frame=cap.read()
    data = pickle.dumps(frame) ### new code
    clientsocket.sendall(struct.pack("L", len(data))+data) ### new code