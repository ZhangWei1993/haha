#!/usr/bin/env python
import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2
import sys
import os
import random
import traceback 
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError
from distutils.version import LooseVersion
from matplotlib import pyplot as plt
from gazebo_msgs.srv import SpawnModel
import tf

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

class GazeboRoom1TurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv._close(self)
        gazebo_env.GazeboEnv.__init__(self, "GazeboRoom1TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=0)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        #randomly spawn the target
        self.add_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.img_rows = 48
        self.img_cols = 64
        self.img_channels = 4
        #TODO this is the list of positions the target box
        self.target_position = np.array([[0,0],[1,1],[2,2]])

        self._seed()

    def obstacle_observation(self,data):
        min_range = 0.2
#        data = np.divide(data, 6.0)
        done = False
#        print(len(data.ranges))
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0.0):
                done = True
        return done

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
        while time.time() < 0.07 + time_1:
            self.vel_pub.publish(vel_cmd)
        self.vel_pub.publish(vel_cmd)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
            except:
                pass
        done = self.obstacle_observation(data)
        cv_image = None
        cv_image = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=10)
        cv_image = CvBridge().imgmsg_to_cv2(cv_image, "bgr8") 
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
#        height, width = cv_image.shape 
#        print ("height : {}, width : {}" .format(height, width))       
        cv_image = cv2.resize(cv_image, (self.img_cols, self.img_rows))
        rospy.wait_for_service('/gazebo/pause_physics', timeout=10)
        try:
            #resp_pause = pause.call()
            self.pause()
#            time_2 = time.time()
#            print(time_2-time_1)
        except Exception as e:
            print ("/gazebo/pause_physics service call failed")
            traceback.print_exc()
        #TODO shaped reward


        if not done:
            if action ==0:
                reward = 0.02
            else:
                reward = 0.01
        else:
            reward = -10
#        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
#        cv2.normalize(cv_image,cv_image,0.0,1.0,cv2.NORM_MINMAX)
#        cv2.imshow('cv_image',cv_image)
#        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        state = cv_image/255.0
        #TODO calculate the polar coordinates of the target in camera frame
        listener = tf.TransformListener()
        try:
            (trans, rot) = listener.lookupTransform('/target', '/camera', rospy.Time(0))
        except Exception as e:
            print("lookupTransform service call failed")
            traceback.print_exc()
        rela_distance = np.sqrt(trans[0]**2+trans[1]**2)
        if trans[0] >=0:
            rela_angle = np.arctan(trans[1] / trans[0])
        else:
            rela_angle = np.arctan(trans[1] / trans[0])+np.pi
        target = np.append(rela_distance,rela_angle)
        vel_state = np.append( vel_cmd.linear.x,vel_cmd.angular.z)
        #        cv_image = cv_image/255.0
#        state = cv_image.reshape(cv_image.shape[0], cv_image.shape[1])
        return state, reward, target, vel_state, done, {}

    def _reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")
        #randomly spawn the target box
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        #TODO change the address and description of the target box,can directly
        model_xml = open("/home/lq/gym-gazebo/gym_gazebo/envs/assets/models/wood_cube_10cm/model.sdf","r").read()
        #TODO change the location of the target box
        pose = Pose()
        # the position of the target, using
        target_no = np.random.choice(self.target_position.shape[0])
        [pose.position.x,pose.position.y] = self.target_position[target_no] + np.random.uniform(-0.2,0.2)
        pose.position.z = 0
        pose.orientation.x = 0
        pose.orientation.y= 0
        pose.orientation.z = 0
        pose.orientation.w = 0
        self.add_model(model_name="target",
                        model_xml=model_xml,
                        robot_namespace="",
                        initial_pose=pose,
                        reference_frame="")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print "/gazebo/unpause_physics service call failed"
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
            except:
                pass
        done = self.obstacle_observation(data)
        cv_image = None
        cv_image = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=10)
        cv_image = CvBridge().imgmsg_to_cv2(cv_image, "bgr8") 
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
#        cv2.namedWindow("window", 1)
#        cv2.imshow("window", cv_image)
        rospy.wait_for_service('/gazebo/pause_physics', timeout=10)
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print "/gazebo/pause_physics service call failed"

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''


#        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_cols, self.img_rows))
        cv_image = cv_image/255.0
        # TODO calculate the polar coordinates of the target in camera frame
        listener = tf.TransformListener()
        try:
            (trans, rot) = listener.lookupTransform('/target', '/camera', rospy.Time(0))
        except Exception as e:
            print("lookupTransform service call failed")
            traceback.print_exc()
        rela_distance = np.sqrt(trans[0] ** 2 + trans[1] ** 2)
        if trans[0] >= 0:
            rela_angle = np.arctan(trans[1] / trans[0])
        else:
            rela_angle = np.arctan(trans[1] / trans[0]) + np.pi
        target = np.append(rela_distance, rela_angle)
#        cv2.imshow('cv_image',cv_image)
        #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image
        return state, target, done
