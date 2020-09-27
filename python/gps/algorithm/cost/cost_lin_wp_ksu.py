""" This file defines an arbitrary linear waypoint cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_LIN_WP_KSU
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evall1l2term, get_ramp_multiplier


class CostLinWPKSU(Cost):
    """ Computes an arbitrary linear waypoint cost. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_LIN_WP_KSU)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T, dU, dX = sample.T, sample.dU, sample.dX

        # Discretize waypoint time steps.
        waypoint_step = np.ceil(T * self._hyperparams['waypoint_time']) # ex. [8., 20.]

        if not isinstance(self._hyperparams['ramp_option'], list):
            self._hyperparams['ramp_option'] = [
                self._hyperparams['ramp_option'] for _ in waypoint_step
            ]

        final_l = np.zeros(T)
        final_lu = np.zeros((T, dU))
        final_lx = np.zeros((T, dX))
        final_luu = np.zeros((T, dU, dU))
        final_lxx = np.zeros((T, dX, dX))
        final_lux = np.zeros((T, dU, dX))

        for data_type in self._hyperparams['data_types']:
            config = self._hyperparams['data_types'][data_type]

            start = 0
            for i in range(len(waypoint_step)):
                wp = config[i]['wp']
                tgt = config[i]['target_state']
                x = sample.get(data_type)
                ##print "\noriginal x:\n", x
                _, dim_sensor = x.shape

                wpm = get_ramp_multiplier(
                    self._hyperparams['ramp_option'][i], int(waypoint_step[i] - start))
                wp = wp * np.expand_dims(wpm, axis=-1)
                
                x = x[start:int(waypoint_step[i]), :]
                ##print "modified x:\n", x
                ##print "target:\n", tgt

                # Compute state penalty.
                dist = x - tgt
                ##print "dist:\n", dist

                # Evaluate penalty term.
                l, ls, lss = evall1l2term(
                    wp, dist, np.tile(np.eye(dim_sensor), [int(waypoint_step[i] - start), 1, 1]),
                    np.zeros((int(waypoint_step[i] - start), dim_sensor, dim_sensor, dim_sensor)),
                    self._hyperparams['l1'], self._hyperparams['l2'],
                    self._hyperparams['alpha']
                )

                final_l[start:int(waypoint_step[i])] = l
                final_lx[start:int(waypoint_step[i]), 0:7] = ls
                final_lxx[start:int(waypoint_step[i]), 0:7, 0:7] = lss

                start = int(waypoint_step[i])
        ##print "Exit on cost lin wp ksu"
        ##exit()
        #return l, lx, lu, lxx, luu, lux
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
