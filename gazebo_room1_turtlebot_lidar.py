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
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
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
        self.path_pub = rospy.Publisher('/path', Path, queue_size=1)
        #randomly spawn the target
        self.add_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.action_space = spaces.Discrete(5) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        #TODO this is the list of positions the target box
        #target for room1
        self.target_position = np.array([[-7.9,6],[-5.6,-0.3],[-10,3.8],[-3.5,-2.8],[-8,1],[-2.9,6.7]])
#        self.map_pub = rospy.Publisher('/map', OccupancyGrid, latch=True)
        self.path = Path()
        self.markerArray = MarkerArray()
        topic = 'visualization_marker_array'
        self.publisher = rospy.Publisher(topic, MarkerArray)
#        self.target_position = np.array([[-7.9,6]])
        #target for esay_level target
#        self.target_position = np.array([[0,4],[0,-3],[-4,0],[3.7,0]])
        #target for empty 
#        self.target_position = np.array([[-1.0,1.0],[1.0,1.0],[1.0,-1.0],[-1.0,-1.0],[0,-2.0],[1,-2.0],[-1,-2.0]])
        self.count=0

        self._seed()
    
    def odom_cb(self,data):
        self.path.header = data.header
        pose = PoseStamped()
        pose.header = data.header
        pose.pose = data.pose.pose
        self.path.poses.append(pose)
        self.path_pub.publish(self.path)
    def to_marker(self, pose):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.pose.position.x = pose.position.x
        marker.pose.position.y = pose.position.x
        marker.pose.position.z = 0

#        quaternion = tf.transformations.quaternion_from_euler(0, 0, self.theta)
#        marker.pose.orientation.x = quaternion[0]
#        marker.pose.orientation.y = quaternion[1]
#        marker.pose.orientation.z = quaternion[2]
#        marker.pose.orientation.w = quaternion[3]

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.2

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.type = Marker.CUBE
        marker.action = marker.ADD
        return marker
    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf'):
                    discretized_ranges.append(6.0)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
#                elif data.ranges[i]>3.0:
#                    discretized_ranges.append(3.0)
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
        self.count = self.count+1
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
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = 0.3
        if action == 4:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.1
            vel_cmd.angular.z = -0.3
#        while time.time() < 0.2 + time_1:
#            self.vel_pub.publish(vel_cmd)
        self.vel_pub.publish(vel_cmd)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
            except:
                pass
        state,done = self.discretize_observation(data,100)
        #TODO shaped reward
        #spawn the robot
        rospy.wait_for_service('/gazebo/get_model_state', timeout=10)        
        in_pose = self.get_model_pose(model_name="mobile_base",relative_entity_name="world").pose
        markrt = self.to_marker(in_pose)
        self.markerArray.markers.append(markrt)
        self.publisher.publish(self.markerArray)
#        model_xml = "<?xml version=\"1.0\"?> \
#                    <robot name=\"myfirst\"> \
#                      <link name=\"world\"> \
#                      </link>\
#                      <link name=\"cylinder0\">\
#                        <visual>\
#                          <geometry>\
#                            <sphere radius=\"0.05\"/>\
#                          </geometry>\
#                          <origin xyz=\"0 0 0\"/>\
#                          <material name=\"rojotransparente\">\
#                              <ambient>0.5 0.5 1.0 0.1</ambient>\
#                              <diffuse>0.5 0.5 1.0 0.1</diffuse>\
#                          </material>\
#                        </visual>\
#                        <inertial>\
#                          <mass value=\"5.0\"/>\
#                          <inertia ixx=\"1.0\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"1.0\" iyz=\"0.0\" izz=\"1.0\"/>\
#                        </inertial>\
#                      </link>\
#                      <joint name=\"world_to_base\" type=\"fixed\"> \
#                        <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\
#                        <parent link=\"world\"/>\
#                        <child link=\"cylinder0\"/>\
#                      </joint>\
#                      <gazebo reference=\"cylinder0\">\
#                        <material>Gazebo/GreenTransparent</material>\
#                      </gazebo>\
#                    </robot>"
#        #TODO change the location of the target box
#        pose = Pose()
#        pose.position.x = in_pose.position.x
#        pose.position.y = in_pose.position.y
#        rospy.wait_for_service('/gazebo/spawn_urdf_model', timeout=20)
#        try: 
#            self.add_model(model_name=str(self.count),
#                            model_xml=model_xml,
#                            robot_namespace="",
#                            initial_pose=pose,
#                            reference_frame="/map")
#        except (rospy.ServiceException) as e:
#            print ("/gazebo/spawn_urdf_model service call failed")
        #spawn the robot
#        odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb)
#        rospy.spin()
#        path_pub = rospy.Publisher('/path', self.path, queue_size=1)
#        pose = PoseStamped()
#        pose.header.stamp = rospy.Time.now()
#        pose.header.frame_id = "odom"
#        pose.pose = in_pose
#        self.path.poses.append(pose)
#        self.path_pub.publish(self.path)
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
                reward = -0.2
            else:
                reward = -0.2
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
#        self.count=0
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
        state,done = self.discretize_observation(data,100)
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
        # for easy level room
        target_no = np.random.choice(self.target_position.shape[0])
        #for HRL use
        pose.position.x = self.target_position[target_no,0]+np.random.uniform(-0.2,0.2)
        pose.position.y = self.target_position[target_no,1]+np.random.uniform(-0.2,0.2)
#        if target_no<2:
#            pose.position.x = self.target_position[target_no,0]+np.random.uniform(-4,3.7)
#            pose.position.y = self.target_position[target_no,1]
#        else:
#            pose.position.x = self.target_position[target_no,0]
#            pose.position.y = self.target_position[target_no,1]+np.random.uniform(-3,4)
#        for empty room
#        pose.position.x = 2
#        pose.position.y = 2.4
        #[pose.position.x,pose.position.y] = self.target_position[target_no] + np.random.uniform(-0.3,0.3)
        print("pose.position.x:{},pose.position.y:{}".format(pose.position.x,pose.position.y))
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