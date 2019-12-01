import cv2
import numpy as np
import socket
import sys
import pickle
import struct ### new code
import imagiz
HOST = '128.32.112.46'
IMG_PORT = 8089
ARR_PORT = 8090


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

def main():
    client = imagiz.Client("cc1",server_ip=HOST, server_port=IMG_PORT)
    vid = cv2.VideoCapture(0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]

    while True:
        try:
            r, frame = vid.read()
            frame = rescale(frame, .5)
            if r:
                r, image = cv2.imencode('.jpg', frame, encode_param)
                client.send(image)
                ### Recieve Array
                data = []
                while True:
                    packet = s.recv(4096)
                    if not packet: break
                    data.append(packet)
                data = pickle.loads(b"".join(data))
                out = pickle.loads(data)
                print(out.shape)
            else:
                break

        except KeyboardInterrupt:
            s.close()
            vid.release()
            cv2.destroyAllWindows()
            break


def rescale(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, ARR_PORT))
    main()