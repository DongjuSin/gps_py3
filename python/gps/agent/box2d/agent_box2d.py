""" This file defines an agent for the Box2D simulator. """
from copy import deepcopy
import numpy as np
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_BOX2D
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, IMAGE_FEAT, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, NOISE
from gps.sample.sample import Sample


## import packages for using realsense camera
import pyrealsense2 as rs
import numpy as np
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os

class AgentBox2D(Agent):
    """
    All communication between the algorithms and Box2D is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = deepcopy(AGENT_BOX2D)
        config.update(hyperparams)
        Agent.__init__(self, config)

        self._setup_conditions()
        self._setup_world(self._hyperparams["world"],
                          self._hyperparams["target_state"],
                          self._hyperparams["render"])

        ## realsense
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
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, world, target, render):
        """
        Helper method for handling setup of the Box2D world.
        """
        self.x0 = self._hyperparams["x0"]
        self._worlds = [world(self.x0[i], target, render)
                        for i in range(self._hyperparams['conditions'])]


    def sample(self, policy, condition, verbose=False, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.

        Args:
            policy: Policy to to used in the trial.
            condition (int): Which condition setup to run.
            verbose (boolean): Whether or not to plot the trial (not used here).
            save (boolean): Whether or not to store the trial into the samples.
            noisy (boolean): Whether or not to use noise during sampling.
        """
        img = []

        self._worlds[condition].run()
        self._worlds[condition].reset_world()
        b2d_X = self._worlds[condition].get_state()
        new_sample = self._init_sample(b2d_X)
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            if (t+1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    self._worlds[condition].run_next(U[t, :])
                b2d_X = self._worlds[condition].get_state()
                print("current step(t): ", t)
                self._set_sample(new_sample, b2d_X, t)
            img_t = new_sample.get(RGB_IMAGE, t)
            img.append(img_t)
        img = np.asarray(img)

        path = '/home/panda_gps/gps_py3/experiments/' + 'box2d_realsense_example' + '/data_files/check_fp'
        if not os.path.exists(path):
            os.mkdir(path)
            print(path, 'is created')
        fname = path + '/fp_%d.npz' % (condition)
        np.savez_compressed(fname, img = img)
        
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)

        return new_sample

    def _init_sample(self, b2d_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, b2d_X, -1)

        feature_fn = None
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

    def _set_sample(self, sample, b2d_X, t):
        for sensor in b2d_X.keys():
            sample.set(sensor, np.array(b2d_X[sensor]), t=t+1)

        feature_fn = None
        if RGB_IMAGE in self.obs_data_types:
            ## TODO : replace below line with panda function
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()            
            img = np.asanyarray(color_frame.get_data())[90:390, 170:470]
            img = np.transpose(img, (2,1,0))
            # cv2.imshow('realsense', self.img)
            # cv2.waitKey()
            
            sample.set(RGB_IMAGE, np.transpose(img, (2, 1, 0)).flatten(), t = t+1)
            ## TODO : check whether below line is neccessary
            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
            if feature_fn is not None:
                obs = sample.get_obs()  # Assumes that the rest of the observation has been populated
                sample.set(IMAGE_FEAT, feature_fn(obs), t=t+1)
            else:
                sample.set(IMAGE_FEAT, np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],)), t=t+1)
