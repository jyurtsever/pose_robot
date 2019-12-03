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
body_parts = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
              'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle',
              'Left Hip', 'Left Knee', 'LAnkle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Background']


def main():
    client = imagiz.Client("cc1",server_ip=HOST, server_port=IMG_PORT)
    vid = cv2.VideoCapture(0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]

    while True:
        try:
            r, frame = vid.read()
            frame = rescale(frame, .75)
            if r:
                r, image = cv2.imencode('.jpg', frame, encode_param)
                client.send(image)
                ### Recieve Array
                data = s.recv(4096)
                points = pickle.loads(data)
                if points:
                    show_points(points, frame)
            else:
                break

        except KeyboardInterrupt:
            s.close()
            vid.release()
            cv2.destroyAllWindows()
            break

def show_points(points, frame):
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    for i, p in enumerate(points):
        if p:
            x, y = p
            cv2.circle(frame, (x, y), inWidth // 50, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(body_parts[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                        lineType=cv2.LINE_AA)

    cv2.imshow("Output-Keypoints", frame)
    cv2.waitKey(2)
    cv2.destroyAllWindows()


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