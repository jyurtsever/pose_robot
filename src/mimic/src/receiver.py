#!/usr/bin/env python
import sys
import numpy as np
import socket
import pickle
import struct ### new code
# sys.path.append('/home/cc/ee106a/fa19/class/ee106a-afo/.local/lib/python3.5/site-packages')
import rospy
from mimic.msg import cache 

HOST = '128.32.112.46'
IMG_PORT = 8098
ARR_PORT = 8097
ARR_PORT_2 = 8099


body_parts = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
              'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle',
              'Left Hip', 'Left Knee', 'LAnkle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Background']

body_dict = {body_parts[i]: i for i in range(len(body_parts))}


def main():
    # Setup ROS publisher node
    rospy.init_node('cache_publisher_node', anonymous=True)
    cache_pub = rospy.Publisher('cache_topic', cache, queue_size = 10)

    joint_dict = {'Neck': None, 'Left Elbow': None, 'Right Wrist': None, 'Left Wrist': None,
            'Right Elbow': None, 'Right Shoulder': None, 'Left Shoulder': None}
    print("hello")
    
    while True:
        try:
            # r, frame = vid.read()
            # frame = rescale(frame, .45)
            # if r:
            # r, image = cv2.imencode('.jpg', frame, encode_param)
            # client.send(image)
            ### Recieve Array
            data = s.recv(4096)
            points = pickle.loads(data)
            if points:
                # show_points(points, frame)
                set_joint_dict(points, joint_dict)
                if all_seen(joint_dict):

                    # Publish cache
                    msg = convert_dict_to_cache(joint_dict)
                    cache_pub.publish(msg)
                    # print('publishing poop')
            # else:
            #     break

        except (KeyboardInterrupt, EOFError) as e:
            s.close()
            vid.release()
            cv2.destroyAllWindows()
            break


def convert_dict_to_cache(joint_dict):
    msg = cache()
    msg.neck = joint_dict['Neck']
    msg.left_shoulder = joint_dict['Left Shoulder']
    msg.right_shoulder = joint_dict['Right Shoulder']
    msg.left_elbow = joint_dict['Left Elbow']
    msg.right_elbow = joint_dict['Right Elbow']
    msg.left_wrist = joint_dict['Left Wrist']
    msg.right_wrist = joint_dict['Right Wrist']
    return msg



"""Sets the necessary body parts in the joint_dict if openpose detects them"""
def set_joint_dict(points, joint_dict):
	for i, pt in enumerate(points):
		if pt and body_parts[i] in joint_dict:
			joint_dict[body_parts[i]] = pt

"""Checks if all the body parts we need have been seen"""
def all_seen(joint_dict):
    for k in joint_dict.keys():
        if not joint_dict[k]:
            return False
    return True

"""Displays Points In Opencv Window"""
# def show_points(points, frame):
#     inWidth = frame.shape[1]
#     inHeight = frame.shape[0]
#     for i, p in enumerate(points):
#         if p:
#             x, y = p
#             cv2.circle(frame, (x, y), inWidth // 50, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
#             cv2.putText(frame, "{}".format(body_parts[i]), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
#                         lineType=cv2.LINE_AA)

#     cv2.imshow("Output-Keypoints", frame)
#     cv2.waitKey(1)
 #   cv2.destroyAllWindows()

"""Recizes image according by the decimal scale (e.g .5)"""
# def rescale(img, scale):
#     width = int(img.shape[1] * scale)
#     height = int(img.shape[0] * scale)
#     dim = (width, height)
#     # resize image
#     resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#     return resized


"""
Moves the robot according to the coordinates stored in the joint_dict. joint_dict is a dictionary mapping the
names of the body parts we need to their points in the frame

"""
def move_joints(joint_dict):
    angles = get_joint_angles_2d(joint_dict)
    #TODO: Finish Implementing get_joint_angels and move robot joints
    

    # print(joint_dict)
    pass


"""Gets the joint angles according to the coordinates stored in the joint_dict
Returns FOUR distinct angles
"""
def get_joint_angles(joint_dict):
    #TODO, IMPLEMENT THIS FUNCTION TO RETURN JOINT ANGELS. Locations of human points given here
    # N = [x,y,z] #neck N
    # S = [x,y,z] #Shoulder S
    # W = [x,y,z] #W
   # Message1.msg
#   Message2.msg
    # n = atan2(norm(cross(S-N,W-N)),dot(S-N,W-N))
    # s = atan2(norm(cross(W-S,N-S)),dot(W-S,N-S))
    # w = atan2(norm(cross(N-W,S-W)),dot(N-W,S-W))

    # a = 
    return None
    #return None

def get_joint_angles_2d(joint_dict):
    #TODO, IMPLEMENT THIS FUNCTION TO RETURN JOINT ANGELS. Locations of human points given here
    # N = [x,y,z] #neck N
    # S = [x,y,z] #Shoulder S
    # W = [x,y,z] #W

    # n = atan2(norm(cross(S-N,W-N)),dot(S-N,W-N))
    # s = atan2(norm(cross(W-S,N-S)),dot(W-S,N-S))
    # w = atan2(norm(cross(N-W,S-W)),dot(N-W,S-W))
    # n = joint_dict['Neck']
    # le = joint_dict['Left Elbow']
    # lw = joint_dict['Left Wrist']
    # ls = joint_dict['Left Shoulder']
    # w = atan2(norm(cross(N-W,S-W)),dot(N-W,S-W))
    print(joint_dict)
    return None
    #return None



if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, ARR_PORT_2))
    main()