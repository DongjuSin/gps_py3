""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np
import rospy
import time

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_PANDA
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, IMAGE_FEAT, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, NOISE

from gps.sample.sample import Sample

from panda_robot import PandaArm
from franka_interface import RobotEnable

class AgentPanda(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_PANDA)
        config.update(hyperparams)
        Agent.__init__(self, config)
        
        rospy.init_node('gps_agent_panda')
        self._setup_conditions()
        # self._setup_world(hyperparams['filename'])
        self._set_initial_state()

        ## TODO : decide to use 'use_camera' or not.
        self.panda = PandaArm()
        # if RGB_IMAGE in self.obs_data_types:
        #     self.panda.use_camera = True

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

    ## NOT CALLED <-- REPLACED TO self.panda._setup_panda_world()

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
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

        ## TODO : where below line should be located?
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

        while True:
            # panda : move to joint position
            print '##### init arm start'
            self.panda.enable_robot()
            self.panda.move_to_joint_position(self._hyperparams['x0'][condition][0:7])
            print '##### init arm finished'
            ## TODO : where below line should be located?
            # new_sample = self._init_sample(condition, feature_fn=feature_fn)   # new_sample: class 'Sample'

            try:    
                # Take the sample.
                for t in range(self.T):
                    X_t = new_sample.get_X(t=t)
                    obs_t = new_sample.get_obs(t=t)
                    mj_U = policy.act(X_t, obs_t, t, noise[t, :])

                    U[t, :] = mj_U
                    # print 'mj_U: ', mj_U
                    # print 'mj_U dict: ', self.list_to_dict(mj_U)
                    
                    if (t + 1) < self.T:
                        # self.panda.enable_robot()
                        # for _ in range(self._hyperparams['substeps']):

                        # panda move with mj_U
                        # self.panda.set_joint_velocities(self.list_to_dict(mj_U))
                        # self.panda.exec_velocity_cmd(mj_U)
                        self.panda.exec_torque_cmd(mj_U)
                        # self.panda.exec_position_cmd(mj_U)

                        if not self.panda.is_enabled_robot():
                            raise StopIteration

                        print "current step(t): ", t
                        self._set_sample(new_sample, mj_X, t, condition, feature_fn=feature_fn)
                break

            except StopIteration:
                print "robot stopped!!!"
                continue
            
            finally:

                f = '/home/panda_gps/gps/experiments/panda_test_dongju/action.npy'
                np.save(f, U)
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

        self.panda.move_to_joint_position(self._hyperparams['x0'][condition][0:7])
        # Initialize sample with stuff from _data
        # panda : get joint positions
        self.prev_positions = self.panda.angles()

        ## TODO : replace below line with panda function
        # get panda joint positions
        sample.set(JOINT_ANGLES, self.prev_positions, t=0)
        # get panda joint velocities
        sample.set(JOINT_VELOCITIES, self.panda.velocities(), t=0)
        # get panda end effector positions
        ee_point, ee_ori = self.panda.ee_pose()
        
        sample.set(END_EFFECTOR_POINTS, ee_point, t=0)
        # get panda end effector velocity
        ee_vel, ee_omg = self.panda.ee_velocity()
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.array(list(ee_vel) + list(ee_omg)), t=0)
        # get panda jacobian
        sample.set(END_EFFECTOR_POINT_JACOBIANS, self.panda.jacobian(), t=0)
        
        ## TODO : check whether below line is neccessary or not.
        if (END_EFFECTOR_POINTS_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINTS_NO_TARGET, np.delete(eepts, self._hyperparams['target_idx']), t=0)
            sample.set(END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, np.delete(np.zeros_like(eepts), self._hyperparams['target_idx']), t=0)
        
        ## TODO : enable this again when after install camera
        '''
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
     	'''
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
        curr_positions = self.panda.angles()

        sample.set(JOINT_ANGLES, curr_positions, t=t+1)
        sample.set(JOINT_VELOCITIES, self.panda.velocities(), t=t+1)
        ee_point, ee_ori = self.panda.ee_pose()
        sample.set(END_EFFECTOR_POINTS, list(ee_point), t=t+1)
        ee_vel, ee_omg = self.panda.ee_velocity()
        sample.set(END_EFFECTOR_POINT_VELOCITIES, list(ee_vel) + list(ee_omg), t=t+1)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, self.panda.jacobian(), t=t+1)

        time_out = True
        s0 = time.time()
        while (np.all(self.prev_positions == curr_positions)):
            s1 = time.time()
            if s1-s0 >= 0.1:
                break
        

        self.prev_positions = curr_positions
        print('Joint Positions: ' + repr(self.prev_positions) + '\n')

        ## TODO : enable this again when after install camera
        '''
        if RGB_IMAGE in self.obs_data_types:
            ## TODO : replace below line with panda function
            self.img = self.baxter.get_baxter_camera_image()
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
		'''

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

    def list_to_dict(self, joint_list):
    	joint_name_list = self.panda.joint_names()
    	joint_dict = {}
    	for i in range(len(joint_list)):
    		joint_dict[joint_name_list[i]] = joint_list[i]
    	return joint_dict