import pickle
import socket
import struct
#from caffemodel2pytorch import caffemodel2pytorch
import caffe
from models import *
import torchvision
import torch.nn as nn
import imagiz
import torch
import cv2

HOST = ''
IMG_PORT = 8099
ARR_PORT = 8098

body_parts = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
              'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle',
              'Left Hip', 'Left Knee', 'LAnkle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Background']

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# print('Socket created')
#
# s.bind((HOST, PORT))
# print('Socket bind complete')
# s.listen(10)
# print('Socket now listening')
#
# conn, addr = s.accept()
#
# data = b'' ### CHANGED
# payload_size = struct.calcsize("L") ### CHANGED
#
# while True:
#
#     # Retrieve message size
#     while len(data) < payload_size:
#         data += conn.recv(4096)
#
#     packed_msg_size = data[:payload_size]
#     data = data[payload_size:]
#     msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED
#
#     # Retrieve all data based on message size
#     while len(data) < msg_size:
#         data += conn.recv(4096)
#
#     frame_data = data[:msg_size]
#     data = data[msg_size:]
#
#     # Extract frame
#     frame = pickle.loads(frame_data)
#     print(frame.shape)
#     # Display
#     # cv2.imshow('frame', frame)
#     # cv2.waitKey(1)
def forward(frame):
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    inpBlob = cv2.dnn.blobFromImage(frame, scalefactor=1.0 / 255, size=(inWidth, inHeight), mean=(0, 0, 0), swapRB=False,
                                   crop=False)

    # Set the prepared object as the input blob of the network
    # print(inpBlob.shape)
    # print(net.blobs)
    net.blobs['image'].reshape(1,3,inHeight,inWidth)
    net.blobs['image'].data[...] = inpBlob
    # net.setInput(inpBlob)

    out = net.forward()
    # inpBlob = torch.from_numpy(inpBlob).cuda()
    # out = model(inpBlob)
    return out['net_output']

def get_points(out, frame):
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]

    H = out.shape[2]
    W = out.shape[3]
    # Empty list to store the detected keypoints
    threshold = .25
    points = []
    for i in range(len(body_parts)):
        # confidence map of corresponding body's part.
        probMap = out[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (inWidth * point[0]) / W
        y = (inHeight * point[1]) / H

        if prob > threshold:
            # cv2.circle(frame, (int(x), int(y)), inWidth // 50, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frame, "{}".format(body_parts[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255),
            #             1,
            #             lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)
    return points

def main():
    print("Connecting...")
    server = imagiz.Server(port=IMG_PORT)
    print("Connected...")
    while True:
        try:
            message = server.receive()
            frame = cv2.imdecode(message.image,1)
            ###Send
            out = forward(frame)
            data_string = pickle.dumps(get_points(out, frame))
            conn.send(data_string)
            cv2.waitKey(1)
        except KeyboardInterrupt:
            s.close()
            cv2.destroyAllWindows()
            break
    print("\nSession Ended")

if __name__ == '__main__':
    # Specify the paths for the 2 files
    # protoFile = 'pose/coco/pose_deploy_linevec.prototxt'
    # weightsFile = 'pose/coco/pose_iter_440000.caffemodel'
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"

    link = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
    pth_file = 'pose/coco/deeppose-COCO.pth'

    # Read the network into Memory
    # net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    print("Initializing Model")
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(protoFile, weightsFile, caffe.TEST)
    # checkpoint = torch.load(pth_file, map_location='cuda:0')['model_state']
    # #print(checkpoint)
    # model = DeepPose(17)
    # model.load_state_dict(checkpoint)
    # model.cuda()
    # model.eval()
    print("Model created")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, ARR_PORT))
    print('Socket bind complete')
    s.listen(1)
    print('Socket now listening')
    conn, addr = s.accept()
    main()
