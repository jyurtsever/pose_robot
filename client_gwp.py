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
    # Serialize frame
    data = pickle.dumps(frame)

    # Send message length first
    message_size = struct.pack("L", len(data)) ### CHANGED

    # Then data
    clientsocket.sendall(message_size + data)