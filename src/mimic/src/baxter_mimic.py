#!/usr/bin/env python

"""
EE106A Final Project 2019: Baxter the Mimicker
Authors: Ben Chang, Joshua Yurtsever, Derek Pan, Hanan Masri
"""

import sys
import rospy
import numpy as np
import baxter_interface
from baxter_interface import Limb
from baxter_interface import CHECK_VERSION
from geometry_msgs.msg import PoseStamped
from mimic.msg import cache

# Mimic Options
mirror = False
manual_control = False

# Global Variables
pose_cache = None

def calculate_angle(vector1, vector2):
    """
    Calculates angle between two vectors using law of cosines
    Returns: angle value (radians) between range 0:pi
    """
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    dot_product = np.dot(vector1, vector2)
    angle = np.arccos(dot_product/(norm1*norm2))
    return angle


def calculate_angle_v2(vector1, vector2):
    """
    Calculates angle between two vectors using arc tangent, more robust than law of cosines
    Returns: angle value (radians)  between range 0:2pi
    """
    angle = np.arctan2(vector2[1],vector2[0]) - np.arctan2(vector1[1],vector1[0])
    return angle


def convert_pose_cache_to_xy_coord():
    """
    Convert OpenPose cache (image pixel coord) to cartesian (XY coordinates) by reflecting across the x axis
    Reflects coordinates on LHS about the y axis to ensure proper angle calculation 
    """
    global pose_cache

    pose = {}
    transform_LHS = np.array([[-1,0], [0,-1]])
    transform_RHS = np.array([[1,0], [0,-1]])

    pose['neck'] = np.dot(np.array(pose_cache.neck), transform_RHS)

    pose['left_shoulder'] = np.dot(np.array(pose_cache.left_shoulder), transform_LHS)
    pose['left_elbow'] = np.dot(np.array(pose_cache.left_elbow), transform_LHS)
    pose['left_wrist'] = np.dot(np.array(pose_cache.left_wrist), transform_LHS)

    pose['right_shoulder'] = np.dot(np.array(pose_cache.right_shoulder), transform_RHS)
    pose['right_elbow'] = np.dot(np.array(pose_cache.right_elbow), transform_RHS)
    pose['right_wrist'] = np.dot(np.array(pose_cache.right_wrist), transform_RHS)

    return pose


def map_cache_to_joint_angles(left_limb, right_limb):
    """
    Converts tracked keypoint coordinates from OpenPose to Baxter Joint Angles
    Desired joint angles to map:
        s0 = shoulder yaw angle             [pi/8: fixed angle for outstretched arms]
        s1 = shoulder abduction angle       [-pi/2]
        e0 = bicep roll angle               [-pi(elbow bends upward) OR 0(elbow bends downward)]
        e1 = elbow angle                    [0(open) to +pi(bicep curled all the way)]
    Returns dictionary of joint angles for both left and right arms
    """
    global pose_cache

    # Do nothing and track current joint angles if no pose_cache has been received from openpose yet
    # or calculations create an exception
    left_angles = left_limb.joint_angles()
    right_angles = right_limb.joint_angles()

    # Negate certain angles when using mirror motion
    mirror_sign = 1
    if mirror:
        mirror_sign = -1

    # Mimic joint control mode
    if pose_cache:
        # Checks that pose_cache is initialized (pose messages has been received from receiver)

        # Convert OpenPose cache (image pixel coord) to cartesian (XY coordinates)
        pose = convert_pose_cache_to_xy_coord()

        # Define OpenPose links
        left_neck_link = pose['neck'] - pose['left_shoulder']                   # vector from shoulder to shoulder                                
        left_upper_arm_link = pose['left_shoulder'] - pose['left_elbow']        # vector from shoulder to elbow
        left_lower_arm_link = pose['left_elbow'] - pose['left_wrist']           # vector from elbow to wrist
        
        right_neck_link = pose['neck'] - pose['right_shoulder']                 # vector from shoulder to shoulder
        right_upper_arm_link = pose['right_shoulder'] - pose['right_elbow']     # vector from shoulder to elbow
        right_lower_arm_link = pose['right_elbow'] - pose['right_wrist']        # vector from elbow to wrist
        
        spine_link = np.array([left_neck_link[1], -left_neck_link[0]])          # spine vector, perpendicular to neck_link
        
        # Preset angles
        left_shoulder_yaw_angle = mirror_sign*np.pi/8
        right_shoulder_yaw_angle = -mirror_sign*np.pi/8

        left_bicep_roll_angle = -mirror_sign*np.pi
        right_bicep_roll_angle = mirror_sign*np.pi

        # Calculate Angles from Links
        try:
            # Calculate abduction and elbow angles
            left_shoulder_abduction_angle = calculate_angle_v2(left_neck_link, left_upper_arm_link)
            right_shoulder_abduction_angle = calculate_angle_v2(right_neck_link, right_upper_arm_link)

            left_elbow_angle = calculate_angle_v2(left_upper_arm_link, left_lower_arm_link) 
            right_elbow_angle = calculate_angle_v2(right_upper_arm_link, right_lower_arm_link)
            
            # Bicep Roll Condition V2 (Bicep faces up or down based on sign of elbow angles)
            bicep_switch_threshold = 0  # switching threshold for biceps to roll to downward position, set to -pi/8 to make less sensitive
            bicep_upper_bound = np.pi
            bicep_lower_bound = np.pi/6
            if left_elbow_angle < bicep_switch_threshold:
                left_bicep_roll_angle = -mirror_sign*bicep_upper_bound
            else:
                left_bicep_roll_angle = -mirror_sign*bicep_lower_bound

            if right_elbow_angle < bicep_switch_threshold:
                right_bicep_roll_angle = mirror_sign*bicep_upper_bound
            else:
                right_bicep_roll_angle = mirror_sign*bicep_lower_bound

            # Take absolute values for elbow angles to constrain range [0, 2pi]
            left_elbow_angle = np.abs(left_elbow_angle)
            right_elbow_angle = np.abs(right_elbow_angle)

            # Set angles 
            left_angles['left_s0'] = left_shoulder_yaw_angle
            left_angles['left_s1'] = left_shoulder_abduction_angle
            left_angles['left_e0'] = left_bicep_roll_angle
            left_angles['left_e1'] = left_elbow_angle

            right_angles['right_s0'] = right_shoulder_yaw_angle
            right_angles['right_s1'] = right_shoulder_abduction_angle
            right_angles['right_e0'] = right_bicep_roll_angle
            right_angles['right_e1'] = right_elbow_angle

            # Debugging printouts (converted to degrees)
            print("left abduction ", left_shoulder_abduction_angle*180/np.pi)
            print("right abduction", right_shoulder_abduction_angle*180/np.pi)

            print("left bicep ", left_bicep_roll_angle*180/np.pi)
            print("right bicep", right_bicep_roll_angle*180/np.pi)

            print("left elbow ", left_elbow_angle*180/np.pi)
            print("right elbow", right_elbow_angle*180/np.pi)

        except:
            print('AN EXCEPTION OCCURRED')
            pass

    # Manual joint control mode (override angle calculations)
    if manual_control:
        # Flex Biceps Pose
        left_angles['left_s0'] = np.pi/8        # left_shoulder_yaw_angle, pi/8 is outstretched
        left_angles['left_s1'] = 0              # left shoulder abduction angle
        left_angles['left_e0'] = np.pi          # left bicep roll angle
        left_angles['left_e1'] = np.pi/2        # left elbow angle
        
        right_angles['right_s0'] = -np.pi/8     # right_shoulder_yaw_angle
        right_angles['right_s1'] = 0            # right shoulder abduction angle
        right_angles['right_e0'] = -np.pi       # right bicep roll angle
        right_angles['right_e1'] = np.pi/2      # right elbow angle

    # Default wrist angles always zero
    left_angles['left_w0'] = 0
    left_angles['left_w1'] = 0
    left_angles['left_w2'] = 0
    right_angles['right_w0'] = 0
    right_angles['right_w1'] = 0
    right_angles['right_w2'] = 0

    # Package left arm and right arm joint angles together
    joint_angles = {'left': left_angles, 'right': right_angles}
    return joint_angles


def mimic_motion(mirror):
    """
    Continously set baxter's joint angles to mimic motion 
    """

    # TODO: check that all the joint angle mappings still work when we set mirror to true 
    if not mirror:
        left_limb = baxter_interface.limb.Limb("left")
        right_limb = baxter_interface.limb.Limb("right")
    else:
        right_limb = baxter_interface.limb.Limb("left")
        left_limb = baxter_interface.limb.Limb("right")

    rate = rospy.Rate(10.0) # 10hz
    
    while not rospy.is_shutdown():
        
        joint_angles = map_cache_to_joint_angles(left_limb, right_limb)
        
        # Set Left Joint Angles, if unchanged use current joint angles
        left_angles = joint_angles['left']
        left_limb.set_joint_positions(left_angles)
        
        # Set Right Joint Angles, if unchanged use current joint angles
        right_angles = joint_angles['right']
        right_limb.set_joint_positions(right_angles)
        
        rate.sleep()


def cache_callback(message):
    """
    Callback function updates global pose_cache variable when new OpenPose cache is received from publisher
    """
    global pose_cache
    pose_cache = message


def main():
    """
    Main Script
    """

    # Initiate Node
    rospy.init_node('mimicker', anonymous=True)
    
    # Enable Robot
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()
    rospy.on_shutdown(clean_shutdown)

    print("Enabling robot... ")
    rs.enable()
    
    # Cache Subscriber - subscribes to the OpenPose Cache dictionary
    print("Subcribing to openpose cache... ")
    rospy.Subscriber('cache_topic', cache, cache_callback)

    # Start the mimicker
    print("Starting Mimicker... ")
    mimic_motion(mirror)


if __name__ == '__main__':
    main()
