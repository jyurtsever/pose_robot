import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import socket
import pickle
import struct ### new code
import imagiz
HOST = '128.32.112.46'
IMG_PORT = 8098
ARR_PORT = 8097


body_parts = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
              'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle',
              'Left Hip', 'Left Knee', 'LAnkle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Background']

body_dict = {body_parts[i]: i for i in range(len(body_parts))}




def main():
    client = imagiz.Client("cc1",server_ip=HOST, server_port=IMG_PORT)
    vid = cv2.VideoCapture(0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    cache = {'Neck': None, 'Left Shoulder': None, 'Right Wrist': None, 'Left Wrist': None,
            'Right Elbow': None, 'Right Shoulder': None, 'Left Shoulder': None}
    while True:
        try:
            r, frame = vid.read()
            frame = rescale(frame, .45)
            if r:
                r, image = cv2.imencode('.jpg', frame, encode_param)
                client.send(image)
                ### Recieve Array
                data = s.recv(4096)
                points = pickle.loads(data)
                if points:
                    show_points(points, frame)
                    set_cache(points, cache)
                    if all_seen(cache):
                    	move_joints(cache)
            else:
                break

        except (KeyboardInterrupt, EOFError) as e:
            s.close()
            vid.release()
            cv2.destroyAllWindows()
            break

"""Sets the necessary body parts in the cache if openpose detects them"""
def set_cache(points, cache):
	for i, pt in enumerate(points):
		if pt and body_parts[i] in cache:
			cache[body_parts[i]] = pt

"""Checks if all the body parts we need have been seen"""
def all_seen(cache):
    for k in cache.keys():
        if not cache[k]:
            return False
    return True

"""Displays Points In Opencv Window"""
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
    cv2.waitKey(1)
 #   cv2.destroyAllWindows()

"""Recizes image according by the decimal scale (e.g .5)"""
def rescale(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


"""
Moves the robot according to the coordinates stored in the cache. Cache is a dictionary mapping the
names of the body parts we need to their points in the frame
"""
def move_joints(cache):
    angles = get_joint_angels(cache)
    #TODO: Finish Implementing get_joint_angels and move robot joints
    # print(cache)
    pass


"""Gets the joint angles according to the coordinates stored in the cache"""
def get_joint_angels(cache):
    #TODO, IMPLEMENT THIS FUNCTION TO RETURN JOINT ANGELS. Locations of human points given here
    return None



if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, ARR_PORT))
    main()