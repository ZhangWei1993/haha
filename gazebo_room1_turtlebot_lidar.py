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
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError
from distutils.version import LooseVersion

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
        self.linear_size = 2
        self.angular_size = 5
        self.action_space = spaces.Discrete(self.linear_size*self.angular_size) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.img_rows = 64
        self.img_cols = 64
        self.img_channels = 1

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
        vel_cmd = Twist()
        vel_cmd.linear.x = (int(action/self.angular_size)+1)*0.2
        vel_cmd.angular.z = (action%self.angular_size-2)*3.1415/12
        while time.time() < 0.2 + time_1:
            self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
            except:
                pass
        done = self.obstacle_observation(data)
        image_data = None
        cv_image = None
        image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=10)
        cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")             

        rospy.wait_for_service('/gazebo/pause_physics', timeout=10)
        try:
            #resp_pause = pause.call()
            self.pause()
#            time_2 = time.time()
#            print(time_2-time_1)
        except Exception as e:
            print ("/gazebo/pause_physics service call failed")
            traceback.print_exc()  


        if not done:
            reward =  vel_cmd.linear.x * np.cos(vel_cmd.angular.z*4)*0.2
        else:
            reward = -10
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        cv_image = cv_image/255.0
        state = cv_image.reshape(cv_image.shape[0], cv_image.shape[1])
        return state, reward, done, {}

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
        image_data = None
        cv_image = None
        image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
        cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print "/gazebo/pause_physics service call failed"

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''


        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        cv_image = cv_image/255.0
        #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(cv_image.shape[0], cv_image.shape[1])
        return state, done
