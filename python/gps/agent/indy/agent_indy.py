""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np
import rospy
import time

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_INDY
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, IMAGE_FEAT, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, NOISE

from gps.sample.sample import Sample

from indy_utils import indydcp_client as client

import pyrealsense2 as rs
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os

class AgentIndy(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_INDY)
        config.update(hyperparams)
        Agent.__init__(self, config)
        
        rospy.init_node('gps_agent_indy')
        self.period = self._hyperparams['dt']
        self.r = rospy.Rate(1. / self.period)
        self._setup_conditions()
        # self._setup_world(hyperparams['filename'])
        self._set_initial_state()

        ## TODO : decide to use 'use_camera' or not.
        robot_ip = “xxx.xxx.x.x”
        robot_name = “xxxx”
        self.indy = client.IndyDCPClient(robot_ip, robot_name)

        # if RGB_IMAGE in self.obs_data_types:
        self.pipeline = rs.pipeline()
        cam_config = rs.config()
        cam_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(cam_config)

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        '''
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
        '''
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _set_initial_state(self):
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            if END_EFFECTOR_POINTS in self.x_data_types:
                '''
                eepts = np.array(self.baxter.get_baxter_end_effector_pose()).flatten()
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts, np.zeros_like(eepts)])
                )
                '''
                self.x0.append(self._hyperparams['x0'][i])
            else:
                self.x0.append(self._hyperparams['x0'][i])
            if IMAGE_FEAT in self.x_data_types:
                self.x0[i] = np.concatenate([self.x0[i], np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],))])
    
    def sample(self, policy, condition, iteration, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        # Create new sample, populate first time step.
        feature_fn = None
        if 'get_features' in dir(policy):
            feature_fn = policy.get_features
        
        new_sample = self._init_sample(condition, feature_fn=feature_fn)
        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)
        
        ### add a function to move indy robot to initial position.
        self.indy.joint_move_to(self._hyperparams[‘x0’][condition][0:6])
        ###
        time.sleep(2)
        
        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
                        
            if (t + 1) < self.T:
                ### add a function that send a torque command to an indy robot.
                self.indy.joint_move_to(mj_U)
                ###

                print("current step(t): ", t)
                self._set_sample(new_sample, mj_X, t, condition, feature_fn=feature_fn)
            
            self.r.sleep() # to sample data at some frequency
        
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)

        if save:
            self._samples[condition].append(new_sample)
        return new_sample
    

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: funciton to comptue image features from the observation.
        """
        sample = Sample(self)

        self.indy.joint_move_to(self._hyperparams[‘x0’][condition][0:6])
        # Initialize sample with stuff from _data
        # indy : get joint positions
        self.prev_positions = self.indy.get_joint_pos()

        ## TODO : replace below line with indy function
        # get indy joint positions
        sample.set(JOINT_ANGLES, self.prev_positions, t=0)

        # get indy joint velocities
        sample.set(JOINT_VELOCITIES, self.indy.get_joint_vel(), t=0)

        # get indy end effector positions
        ee_point = self.indy.get_task_pos()[:3]
        sample.set(END_EFFECTOR_POINTS, ee_point, t=0)
        # sample.set(END_EFFECTOR_POINTS, list(ee_point), t=t+1)

        # get indy end effector velocity
        vel = self.indy.get_task_vel()
        ee_vel = vel[:3]
        ee_omg = vel[3:]
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.array(list(ee_vel) + list(ee_omg)), t=0)

        # get indy jacobian
        ### please add a function that retreive jacobian matrix here.
        sample.set(END_EFFECTOR_POINT_JACOBIANS, self.indy, t=0)
        
        ## TODO : check whether below line is neccessary or not.
        if (END_EFFECTOR_POINTS_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINTS_NO_TARGET, np.delete(eepts, self._hyperparams['target_idx']), t=0)
            sample.set(END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, np.delete(np.zeros_like(eepts), self._hyperparams['target_idx']), t=0)
        
        ## TODO : enable this again when after install camera
        
        # only save subsequent images if image is part of observation
        if RGB_IMAGE in self.obs_data_types:
            ## TODO : replace below line with other function
            # ex 1:
            # self.img = self.baxter.get_baxter_camera_image()
            # sample.set(RGB_IMAGE, np.transpose(self.img, (2, 1, 0)).flatten(), t = 0)
            # ex 2:
            # sample.set(RGB_IMAGE, img_data, t=0)

            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
            if IMAGE_FEAT in self.obs_data_types:
                raise ValueError('Image features should not be in observation, just state')
            if feature_fn is not None:
                obs = sample.get_obs()  # Assumes that the rest of the sample has been populated
                sample.set(IMAGE_FEAT, feature_fn(obs), t=0)
            else:
                sample.set(IMAGE_FEAT, np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],)), t=0)
     	
        return sample

    def _set_sample(self, sample, mj_X, t, condition, feature_fn=None):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """
        curr_positions = self.indy.get_joint_pos()

        sample.set(JOINT_ANGLES, curr_positions, t=t+1)
        sample.set(JOINT_VELOCITIES, self.indy.get_joint_vel(), t=t+1)
        ee_point = self.indy.get_task_pos()[:3]
        sample.set(END_EFFECTOR_POINTS, ee_point, t=t+1)
        # sample.set(END_EFFECTOR_POINTS, list(ee_point), t=t+1)
        vel = self.indy.get_task_vel()
        ee_vel = vel[:3]
        ee_omg = vel[3:]
        sample.set(END_EFFECTOR_POINT_VELOCITIES, list(ee_vel) + list(ee_omg), t=t+1)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, self.indy, t=t+1)
        
        time_out = True
        """
        s0 = time.time()
        while (np.all(self.prev_positions == curr_positions)):
            s1 = time.time()
            if s1-s0 >= 0.1:
                break
        """
        self.prev_positions = curr_positions
        print('Joint Positions: ' + repr(self.prev_positions) + '\n')

        ## TODO : enable this again when after install camera
        
        if RGB_IMAGE in self.obs_data_types:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()            
            img = np.asanyarray(color_frame.get_data())[90:390, 170:470]
            img = np.transpose(img, (2,1,0))
            
            sample.set(RGB_IMAGE, np.transpose(self.img, (2, 1, 0)).flatten(), t = t+1)
            ## TODO : check whether below line is neccessary
            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
            if feature_fn is not None:
                obs = sample.get_obs()  # Assumes that the rest of the observation has been populated
                sample.set(IMAGE_FEAT, feature_fn(obs), t=t+1)
            else:
                sample.set(IMAGE_FEAT, np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],)), t=t+1)
		
    def _get_image_from_obs(self, obs):
        imstart = 0
        imend = 0
        image_channels = self._hyperparams['image_channels']
        image_width = self._hyperparams['image_width']
        image_height = self._hyperparams['image_height']
        for sensor in self._hyperparams['obs_include']:
            # Assumes only one of RGB_IMAGE or CONTEXT_IMAGE is present
            if sensor == RGB_IMAGE or sensor == CONTEXT_IMAGE:
                imend = imstart + self._hyperparams['sensor_dims'][sensor]
                break
            else:
                imstart += self._hyperparams['sensor_dims'][sensor]
        img = obs[imstart:imend]
        img = img.reshape((image_width, image_height, image_channels))
        ## TODO : check whether below line is neccessary
        img = img.astype(np.uint8)
        return img