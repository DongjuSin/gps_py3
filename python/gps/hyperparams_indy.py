""" Hyperparameters for Baxter peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.indy.agent_indy import AgentIndy

#from gps.algorithm.cost.cost_lin_wp import CostLinWP
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
# from gps.algorithm.cost.cost_lin_wp_ksu import CostLinWPKSU
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT, RAMP_QUADRATIC, RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from gps.gui.config import generate_experiment_info

from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_example import tf_network, multi_modal_network, multi_modal_network_fp 
#from gps.algorithm.cost.cost_lin_wp import CostLinWp

DOF = 6
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_CHANNELS = 300
SENSOR_DIMS = {
    JOINT_ANGLES: DOF,
    JOINT_VELOCITIES: DOF,
    END_EFFECTOR_POINTS: 3, #6
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: DOF,
    
}

# PR2_GAINS = np.array([0.4, 0.5, 0.5, 0.6, 0.5, 0.6, 0.6])
# PR2_GAINS = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
PR2_GAINS = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/indy_example/'


CONDITIONS = 1
np.random.seed(14)
pos_body_offset = []
#for _ in range(CONDITIONS):
#    pos_body_offset.append([np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1, 0])])

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    
    'conditions': CONDITIONS,
    'train_conditions': range(CONDITIONS),
    'test_conditions': range(CONDITIONS),

    #'no_sample_logging': True,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

# [JOINT_ANGLES(6), JOINT_VELOCITIES(6), END_EFFECTOR_POINTS(3) (x,y,z), END_EFFECTOR_POINT_VELOCITIES(6)]
x0s = [np.array(['fill here', 'fill here', 'fill here', 'fill here', 'fill here', 'fill here',
                 0., 0., 0., 0., 0., 0., 0.,
                 'fill here', 'fill here', 'fill here',
                 0., 0., 0., 0., 0., 0.])]


# initial_left_arm = [np.array([-0.5042961834411621, 0.4536748174987793, -2.2212041783203125, -0.04908738515625, 0.4406359808166504, 2.0248546376953125, 0.1928980838562012])]



agent = {
    'type': AgentIndy,
    #'filename': './mjc_models/pr2_arm3d.xml',
    'x0': x0s,
    #'initial_left_arm': initial_left_arm,
    'dt': 0.05, #4. #0.05 (original),
    #'substeps': 5,
    'conditions': common['conditions'],
    #'pos_body_idx': np.array([1]),
    #'pos_body_offset': [np.array([0, -0.2, 0]) for i in range(common['conditions'])],
    'T': 50, # 300,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES], ## END_EFFECTOR MAY BE INCLUDED
    
}

algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'iterations': 40,

    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],

    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS,
}

state_cost = [{
    'type': CostState,
    'data_types': {
        JOINT_ANGLES: {
            'wp': np.ones(6),
            'target_state': np.array(['fill here', 'fill here', 'fill here', 'fill here', 'fill here', 'fill here']),
        },
    },
    'ramp_option': RAMP_QUADRATIC,
} for i in xrange(common['conditions'])]

'''
linwp_cost = [{
    'type': CostLinWPKSU,
    'waypoint_time': np.array([0.33, 0.66, 1.]),
    'data_types': {
        JOINT_ANGLES: [
            {
                'wp': np.ones(6),
                'target_state': np.array(['fill here', 'fill here', 'fill here', 'fill here', 'fill here', 'fill here']),
            },
            {
                'wp': np.ones(6),
                'target_state': np.array(['fill here', 'fill here', 'fill here', 'fill here', 'fill here', 'fill here']),
            },
            {
                'wp': np.ones(6),
                'target_state': np.array(['fill here', 'fill here', 'fill here', 'fill here', 'fill here', 'fill here']),
            },
        ],
    },
    'ramp_option': [RAMP_QUADRATIC, RAMP_QUADRATIC, RAMP_QUADRATIC, RAMP_QUADRATIC, RAMP_QUADRATIC, RAMP_QUADRATIC],
} for i in xrange(common['conditions'])]
'''
algorithm['cost'] = [{
    'type': CostSum,
    'costs': [torque_cost, state_cost[i]],
    'weights': [0.1, 2.0],
} for i in xrange(common['conditions'])]

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'num_filters': [64, 32, 32],
        'obs_include': agent['obs_include'],
        'obs_vertor_data': [JOINT_ANGLES, JOINT_VELOCITIES],
        'obs_image_data': [], #[RGB_IMAGE],
        
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
    },
    #'network_model': multi_modal_network,
    'network_model': tf_network,
    'iterations': 1000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

num_samples = 5
config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': num_samples,
    'verbose_policy_trials': 1,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': num_samples, #5,
}

common['info'] = generate_experiment_info(config)

