#!/usr/bin/env python

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Joint Position Example: keyboard
"""
import argparse

import rospy

import baxter_interface
import baxter_external_devices

from baxter_interface import CHECK_VERSION
import sys


def move_robot(target_thetas):
 	
# s0 = shoulder yaw (+ opens)
# s1 = shoulder ab/adduction (+lowers)
# e0 = bicep roll (+CW from Baxter's perspective)
# e1 = elbow (+curl)
# 2-D orientation: i.e. 0.4 0 -3.1 2 0 0 0
    r = rospy.Rate(10)
    joint_command = {'left_s0': target_thetas[0], 'left_s1': target_thetas[1], 'left_e0': target_thetas[2], 'left_e1': target_thetas[3], 'left_w0': target_thetas[4], 'left_w1': target_thetas[5], 'left_w2': target_thetas[6]}
 	# joint_command = {'left_s0': target_thetas[0], 'left_s1': target_thetas[1], 'left_e0': target_thetas[2], 'left_e1': target_thetas[3], 'left_w0': target_thetas[4], 'left_w1': target_thetas[5], 'left_w2': target_thetas[6]}
    # print joint_command
	# joint_command = {'left_s0': target_thetas[0], 'left_s1': target_thetas[1], 'left_e0': target_thetas[2], 'left_e1': target_thetas[3], 'left_w0': target_thetas[4], 'left_w1': target_thetas[5], 'left_w2': target_thetas[6]}
	# print joint_command
	# done = False
 #    print("Controlling joints. Press ? for help, Esc to quit.")

    left = baxter_interface.Limb('left')

    while not rospy.is_shutdown():


	    left.set_joint_positions(joint_command)
	    # i=-1
	    # compare = []
	    # for joint in joint_command:
	    # 	i=i+1
	    # 	current_position = left.joint_angle(joint)
	    # 	comparei = abs(current_position) - abs(target_thetas[i])
	    # 	compare.append(comparei)
	    # print compare
	    # if max(compare) < 0.01:
	    # 		# done = True
	    # 	break
	    # if done:
	    # 	break
	   


	  	

	



	    
	    r.sleep()


def main():
    """RSDK Joint Position Example: Keyboard Control

    Use your dev machine's keyboard to control joint positions.

    Each key corresponds to increasing or decreasing the angle
    of a joint on one of Baxter's arms. Each arm is represented
    by one side of the keyboard and inner/outer key pairings
    on each row for each joint.
    """
    epilog = """
See help inside the example with the '?' key for key bindings.
    """
    

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_position_keyboard")
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

    # map_keyboard()
    print("Done.")

	# if not len(sys.argv) == 8:
 #    	raise TypeError('Must input a 7-vector')
    target_thetas = [float(sys.argv[1]),
    				float(sys.argv[2]),
    				float(sys.argv[3]), 
    				float(sys.argv[4]), 
    				float(sys.argv[5]), 
    				float(sys.argv[6]),
    				float(sys.argv[7])]			
    
    print target_thetas
    # source_frame = sys.argv[2]
    move_robot(target_thetas)


# https://stackoverflow.com/questions/17118999/python-argparse-unrecognized-arguments
if __name__ == '__main__':
    main()

