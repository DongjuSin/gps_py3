import sys
import numpy as np
from matplotlib import pyplot as plt

base_dir = '/hdd/gps-master/experiments/'
exp_name = 'state_only'
itr = int(sys.argv[1])
num_cond = 1

initial = np.array([0.6893729294751658, -0.27719589272540046])
target = np.array([[0.7279655652628775, 0.03726558015192617], [0.8123950797614842, 0.05028600764771304]])

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

axarr = []
axarr.append(ax1)
axarr.append(ax2)

for i in range(2):
	axarr[i].plot(initial[0], initial[1], '^')
	axarr[i].plot(target[0, 0], target[0, 1], '^')
	axarr[i].plot(target[1, 0], target[1, 1], '^')

for i in range(num_cond):
	f_con = 'controller_%d_%d.npz' % (itr, i)
	f_pol = 'policy_%d_%d.npz' % (itr, i)
	
	con = np.load(base_dir + exp_name + '/data_files/check_traj/' + f_con)
	pol = np.load(base_dir + exp_name + '/data_files/check_traj/' + f_pol)
	
	con_eept = con['mu_eept']
	pol_eept = pol['ee_pt']

	axarr[0].plot(con_eept[:,0], con_eept[:,1], '.')
	axarr[1].plot(pol_eept[:,0], pol_eept[:,1], '.')

axarr[0].set_title('local controller trajectory')
axarr[1].set_title('policy trajectory')
for i in range(2):	
	axarr[i].set_xlim(0.65, 0.85)
	axarr[i].set_ylim(-0.35, 0.2)
	axarr[i].set_xlabel('x (m)', fontsize = 14)
	axarr[i].set_ylabel('y (m)', fontsize = 14)
axarr[1].legend(['initial', 'target_0', 'target_1', 'traj_0', 'traj_1'], bbox_to_anchor=(1.2, 1))
# axarr[1].legend(['initial', 'target_0', 'target_1', 'traj_0', 'traj_1'])
axarr[1].yaxis.tick_right()

path = base_dir + exp_name + '/data_files/check_traj/'
# if not os.path.exists(path):
	# os.mkdir(path)
	# print path, ' is created'
fname = path + 'itr_%d.png' % itr
plt.savefig(fname, bbox_inches='tight')
plt.close(fig)
# plt.show()
'''

edges = 100

con = np.load(base_dir + exp_name + '/data_files/' + 'controller_0_0.npz')
mu = con['mu_eept']
sigma = con['sigma_eept']

p = np.linspace(0, 2*np.pi, edges)
xy_ellipse = np.c_[np.cos(p), np.sin(p)]
T = mu.shape[0]

sigma_xy = sigma[:, 0:2, 0:2]
u, s, v = np.linalg.svd(sigma_xy)

for t in range(T):
	xyz = np.repeat(mu[t, :].reshape((1, 3)), edges, axis=0)
	xyz[:, 0:2] += np.dot(xy_ellipse, np.dot(np.diag(np.sqrt(s[t, :])), u[t, :, :].T))
	print np.sqrt(u[t,0,0]**2 + u[t,0,1])
	print np.sqrt(u[t,1,0]**2 + u[t,1,1])
	plt.plot(xyz[:,0], xyz[:,1])
plt.xlim(0.5, 1.0)
plt.ylim(-0.25, 0.05)
plt.axes().set_aspect('equal')
plt.show()
'''