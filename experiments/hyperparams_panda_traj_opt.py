""" Hyperparameters for Baxter peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.panda.agent_panda import AgentPanda

#from gps.algorithm.cost.cost_lin_wp import CostLinWP
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_lin_wp_ksu import CostLinWPKSU
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT, RAMP_QUADRATIC, RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
#from gps.algorithm.traj_opt.
from gps.algorithm.policy.lin_gauss_init import init_lqr

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from gps.gui.config import generate_experiment_info
##

from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_example import tf_network, multi_modal_network, multi_modal_network_fp 
#from gps.algorithm.cost.cost_lin_wp import CostLinWp


IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IMAGE_CHANNELS = 300
SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 3, #6
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 7,
    
}

# PR2_GAINS = np.array([0.4, 0.5, 0.5, 0.6, 0.5, 0.6, 0.6])
# PR2_GAINS = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
# PR2_GAINS = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
# PR2_GAINS = np.array([393.43, 36.46, 71.35, 2.500, 2.528, 1.067, 5.099])
# PR2_GAINS = np.flip(np.array([393.43, 36.46, 71.35, 2.500, 2.528, 1.067, 5.099]))
# PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])
PR2_GAINS = np.array([10000, 10000, 10000, 10000, 10000, 10000, 10000])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/panda_traj_opt/'


CONDITIONS = 1
np.random.seed(6)
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

# [JOINT_ANGLES(7), JOINT_VELOCITIES(7), END_EFFECTOR_POINTS(3) (x,y,z), END_EFFECTOR_POINT_VELOCITIES(6)]
'''
x0s = [np.array([ 6.95359905e-04, -7.26338674e-01, -9.45553346e-04, -2.33732858e+00,
 -1.52128660e-03,  1.57128737e+00,  7.21282021e-01,
                 0., 0., 0., 0., 0., 0., 0.,
                  3.22498757e-01, -3.21665427e-04,  5.80631059e-01,
                 0., 0., 0., 0., 0., 0.])]
'''
x0s = [np.array([-0.21352394, -1.14777156, -0.12204788, -2.55762594, -0.12411306, 1.42176939, 0.46069221,
                 0., 0., 0., 0., 0., 0., 0.,
                  3.22498757e-01, -3.21665427e-04,  5.80631059e-01,
                 0., 0., 0., 0., 0., 0.])]


# initial_left_arm = [np.array([-0.5042961834411621, 0.4536748174987793, -2.2212041783203125, -0.04908738515625, 0.4406359808166504, 2.0248546376953125, 0.1928980838562012])]



agent = {
    'type': AgentPanda,
    #'filename': './mjc_models/pr2_arm3d.xml',
    'x0': x0s,
    #'initial_left_arm': initial_left_arm,
    'dt':  0.05, #0.05, #4. #0.05 (original),
    #'substeps': 5,
    'conditions': common['conditions'],
    #'pos_body_idx': np.array([1]),
    #'pos_body_offset': [np.array([0, -0.2, 0]) for i in range(common['conditions'])],
    'T': 20, # 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES], ## END_EFFECTOR MAY BE INCLUDED
    
}
'''
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
'''
algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 40,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': 1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 0.5, # 1.0
    'stiffness': 0.5, # 1.0,
    'stiffness_vel': 0.25, # 0.5,
    'final_weight': 50.0, # 50,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS,
}
'''
fk_cost1 = {
    'type': CostFK,
    # Target end effector is subtracted out of EE_POINTS in ROS so goal
    # is 0.
    'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 0.0001,
    'ramp_option': RAMP_LINEAR,
}

fk_cost2 = {
    'type': CostFK,
    'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 0.0,
    'wp_final_multiplier': 10.0,  # Weight multiplier on final timestep.
    'ramp_option': RAMP_FINAL_ONLY,
}
'''
state_cost = [{
    'type': CostState,
    'data_types': {
        JOINT_ANGLES: {
            'wp': np.ones(7),
            # 'target_state': np.array([-0.58546234, 0.12736591, 0.8894016, -2.57350823, -0.21100376, 2.63750599, 0.59722294]),
	    'target_state': np.array([-0.22550496, -0.6364437, 0.52940194, -2.26652302, 0.26930142, 1.70780718, 0.64340339]),
        },
    },
    'ramp_option': RAMP_QUADRATIC,
} for i in xrange(common['conditions'])]


linwp_cost = [{
    'type': CostLinWPKSU,
    'waypoint_time': np.array([0.33, 0.66, 1.]),
    'data_types': {
        JOINT_ANGLES: [
            {
                'wp': np.ones(7),
                'target_state': np.array([-0.19469054006333333, -0.4417704793333333, 0.2958368311026667, -2.4160551299999997, -0.07134877773333334, 1.9266935766666669, 0.6799289940000001]),
            },
            {
                'wp': np.ones(7),
                'target_state': np.array([-0.39007644003166664, -0.15720228466666664, 0.5926192155513333, -2.49478168, -0.14117626886666668, 2.2820997833333334, 0.638575967]),
            },
            {
                'wp': np.ones(7),
                'target_state': np.array([-0.58546234, 0.12736591, 0.8894016, -2.57350823, -0.21100376,  2.63750599, 0.59722294]),
            },
        ],
    },
    'ramp_option': [RAMP_QUADRATIC, RAMP_QUADRATIC, RAMP_QUADRATIC, RAMP_QUADRATIC, RAMP_QUADRATIC, RAMP_QUADRATIC],
} for i in xrange(common['conditions'])]

algorithm['cost'] = [{
    'type': CostSum,
    # 'costs': [torque_cost, state_cost[i]],
    'costs': [torque_cost, linwp_cost[i], state_cost[i]],
    #'costs': [torque_cost, state_cost_1[i], state_cost_2[i], linwp_cost_1[i], linwp_cost_2[i]],
    'weights': [1.0, 1.0, 3.0], # [10.0, 2.0, 3.0],
    
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
'''
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
    'iterations': 3000,
    'weights_file_prefix': EXP_DIR + 'policy',
}
'''
algorithm['policy_opt'] = {}
'''
algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}
'''
num_samples = 10
config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 0, # num_samples,
    'verbose_policy_trials': 1,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': num_samples, #5,
}

common['info'] = generate_experiment_info(config)

