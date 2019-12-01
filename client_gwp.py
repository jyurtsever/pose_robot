import cv2
import numpy as np
import socket
import sys
import pickle
import struct ### new code
import imagiz
HOST = '128.32.112.46'
PORT = 8089


# cap=cv2.VideoCapture(0)
# clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# clientsocket.connect((HOST, PORT))
# while True:
#     ret,frame=cap.read()
#     # Serialize frame
#     data = pickle.dumps(frame)
#
#     # Send message length first
#     message_size = struct.pack("L", len(data)) ### CHANGED
#
#     # Then data
#     clientsocket.sendall(message_size + data)

client=imagiz.Client("cc1",server_ip=HOST)
vid=cv2.VideoCapture(0)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    r,frame=vid.read()
    if r:
        r, image = cv2.imencode('.jpg', frame, encode_param)
        client.send(image)
    else:
        break
