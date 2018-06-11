#!/usr/bin/env python
import gym
import rospy
import roslib
import roslaunch
import time
import numpy as np
import cv2
import sys
import os
import random
import traceback 
import subprocess
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from distutils.version import LooseVersion
from gazebo_msgs.srv import SpawnModel,DeleteModel,GetModelState
from tf.transformations import euler_from_quaternion
#import tf
import rosnode

class GazeboRoom1TurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv._close(self)
        gazebo_env.GazeboEnv.__init__(self, "GazeboRoom1TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=0)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.get_model_pose = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        #randomly spawn the target
        self.add_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.action_space = spaces.Discrete(5) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        #TODO this is the list of positions the target box
#        self.target_position = np.array([[-1.2,1.2],[1.2,1.2],[1.2,-1.2],[-6,3.8],[0.36,4.6],[4.3,1.6],[7.7,4.8],[6.5,1.5],[6.8,-1.4],[-6.3,-1.7]])
        self.target_position = np.array([[-1.0,1.0],[1.0,1.0],[1.0,-1.0],[-1.0,-1.0],[0,-2.0],[1,-2.0],[-1,-2.0]])

        self._seed()


    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf'):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(data.ranges[i])
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics', timeout=10)
        try:
            self.unpause()
            time_1 = time.time()
        except Exception as e:
            print "/gazebo/unpause_physics service call failed"
            traceback.print_exc()
        if action == 0:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        if action == 1:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = 0.2
        if action == 2:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.2
            vel_cmd.angular.z = -0.2
        if action == 3:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.2
        if action == 4:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.2
        while time.time() < 0.2 + time_1:
            self.vel_pub.publish(vel_cmd)
        self.vel_pub.publish(vel_cmd)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
            except:
                pass
        state,done = self.discretize_observation(data,50)
        #TODO shaped reward
        #TODO calculate the polar coordinates of the target in camera frame
        rospy.wait_for_service('/gazebo/get_model_state', timeout=10)        
        in_pose = self.get_model_pose(model_name="mobile_base",relative_entity_name="world").pose
        abs_x = self.target_pose.position.x - in_pose.position.x
        abs_y = self.target_pose.position.y - in_pose.position.y
        (roll, pitch, yaw) = euler_from_quaternion ([in_pose.orientation.x,in_pose.orientation.y,in_pose.orientation.z,in_pose.orientation.w])
        trans_matrix = np.matrix([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        rela = np.matmul(trans_matrix,np.array([[abs_x],[abs_y]]))
        rela_x = rela[0,0]
        rela_y = rela[1,0]
#        ads_dis = np.sqrt(abs_x**2+abs_y**2)
        rela_distance = np.sqrt(rela_x** 2 + rela_y ** 2)
#        print("distance: {}" .format(ads_dis))
#        rela_distance = np.sqrt(rela_x** 2 + rela_y ** 2)
        if rela_x >= 0:
            rela_angle = np.arctan(rela_y / (rela_x+ 0.00000001))
        else:
            rela_angle = np.arctan(rela_y / (rela_x- 0.00000001)) + np.pi
        target = np.append(rela_distance, rela_angle)
        vel_state = np.append( vel_cmd.linear.x,vel_cmd.angular.z)
        if not done:
            if action ==0:
                reward = -0.1
            elif action ==1 or action ==2:
                reward = -0.1
            else:
                reward = -0.1
        else:
            reward = -10
        if rela_distance <= 0.4:
            done = True
            reward = 10
            # delete the spwaned model at last time
#        if done:
#            rospy.wait_for_service('/gazebo/delete_model', timeout=10)
#            try:
#                #reset_proxy.call()
#                self.delete_model(model_name="target")
#            except (rospy.ServiceException) as e:
#                print ("/gazebo/delete_model service call failed")  
        rospy.wait_for_service('/gazebo/pause_physics', timeout=10)
        try:
            #resp_pause = pause.call()
            self.pause()
#            time_2 = time.time()
#            print(time_2-time_1)
        except Exception as e:
            print ("/gazebo/pause_physics service call failed")
            traceback.print_exc()
        return state, reward, target, vel_state, done

    def _reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print "/gazebo/unpause_physics service call failed"
        rospy.wait_for_service('/gazebo/delete_model', timeout=10)
        try:
            #reset_proxy.call()
            self.delete_model(model_name="target")
        except (rospy.ServiceException) as e:
            print ("/gazebo/delete_model service call failed")  
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
            except:
                pass
        state,done = self.discretize_observation(data,50)
        #randomly spawn the target box
        #TODO change the address and description of the target box,can directly
        model_xml = "<?xml version=\"1.0\"?> \
                    <robot name=\"myfirst\"> \
                      <link name=\"world\"> \
                      </link>\
                      <link name=\"cylinder0\">\
                        <visual>\
                          <geometry>\
                            <sphere radius=\"0.3\"/>\
                          </geometry>\
                          <origin xyz=\"0 0 0\"/>\
                          <material name=\"rojotransparente\">\
                              <ambient>0.5 0.5 1.0 0.1</ambient>\
                              <diffuse>0.5 0.5 1.0 0.1</diffuse>\
                          </material>\
                        </visual>\
                        <inertial>\
                          <mass value=\"5.0\"/>\
                          <inertia ixx=\"1.0\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"1.0\" iyz=\"0.0\" izz=\"1.0\"/>\
                        </inertial>\
                      </link>\
                      <joint name=\"world_to_base\" type=\"fixed\"> \
                        <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\
                        <parent link=\"world\"/>\
                        <child link=\"cylinder0\"/>\
                      </joint>\
                      <gazebo reference=\"cylinder0\">\
                        <material>Gazebo/RedTransparent</material>\
                      </gazebo>\
                    </robot>"
        #TODO change the location of the target box
        pose = Pose()
        # the position of the target, using
        target_no = np.random.choice(self.target_position.shape[0])
        [pose.position.x,pose.position.y] = self.target_position[target_no] + np.random.uniform(-0.4,0.4)
#        [pose.position.x,pose.position.y] = self.target_position[target_no]
        pose.position.z = 0.5
        pose.orientation.x = 0
        pose.orientation.y= 0
        pose.orientation.z = 0
        pose.orientation.w = 0
        self.target_pose = pose
        rospy.wait_for_service('/gazebo/spawn_urdf_model', timeout=10)
        try: 
            self.add_model(model_name="target",
                            model_xml=model_xml,
                            robot_namespace="",
                            initial_pose=pose,
                            reference_frame="/map")
        except (rospy.ServiceException) as e:
            print ("/gazebo/spawn_urdf_model service call failed")
        #get the pose of the target with repect to the camera
        rospy.wait_for_service('/gazebo/get_model_state', timeout=10)
        in_pose = self.get_model_pose(model_name="mobile_base",relative_entity_name="world").pose
        abs_x = self.target_pose.position.x - in_pose.position.x
        abs_y = self.target_pose.position.y - in_pose.position.y
        (roll, pitch, yaw) = euler_from_quaternion ([in_pose.orientation.x,in_pose.orientation.y,in_pose.orientation.z,in_pose.orientation.w])
        trans_matrix = np.matrix([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        rela = np.matmul(trans_matrix,np.array([[abs_x],[abs_y]]))
        rela_x = rela[0,0]
        rela_y = rela[1,0]
#        ads_dis = np.sqrt(abs_x**2+abs_y**2)
        rela_distance = np.sqrt(rela_x** 2 + rela_y ** 2)
        if rela_x >= 0:
            rela_angle = np.arctan(rela_y / (rela_x+ 0.00000001))
        else:
            rela_angle = np.arctan(rela_y / (rela_x- 0.00000001)) + np.pi
        target = np.append(rela_distance, rela_angle)
        #print the distance and angle of the target point
#        print("distance: {}" .format(ads_dis))
        rospy.wait_for_service('/gazebo/pause_physics', timeout=10)
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print "/gazebo/pause_physics service call failed"
        
        # TODO calculate the polar coordinates of the target in camera frame
#        subprocess.Popen("rosrun tf view_frames")
        return state, target, done