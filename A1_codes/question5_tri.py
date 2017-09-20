import numpy as np
import plotly.plotly as py
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from plotly.graph_objs import *
import plotly.offline as ploo
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.cm as cm

def BM_formula(x1, x2):
	y1 = ((-2*np.log(x1))**(0.5))*np.cos(2*np.pi*x2)
	y2 = ((-2*np.log(x1))**(0.5))*np.sin(2*np.pi*x2)
	return y1, y2

def BM_sampling_method(num):
	uni1 = np.random.uniform(low=0.0, high=1.0, size=num)
	uni2 = np.random.uniform(low=0.0, high=1.0, size=num)
	lst1 = []
	lst2 = []
	for i in range(len(uni1)):
		a, b = BM_formula(uni1[i], uni2[i])
		lst1.append(a)
		lst2.append(b)

	return np.asarray([np.asarray(lst1), np.asarray(lst2)])

def experiment_res(x_list, y_list):
    """ An analytic function representing experiment results """
    z_list = []
    for i in range(len(x_list)):
      x = x_list[i]
      y = y_list[i]
      #print type(x), type(y)
      x = 2.*x
      r1 = np.sqrt((0.5 - x)**2 + (0.5 - y)**2)
      theta1 = np.arctan2(0.5 - x, 0.5 - y)
      r2 = np.sqrt((-x - 0.2)**2 + (-y - 0.2)**2)
      theta2 = np.arctan2(-x - 0.2, -y - 0.2)
      z = (4*(np.exp((r1/10)**2) - 1)*30. * np.cos(3*theta1) +
          (np.exp((r2/10)**2) - 1)*30. * np.cos(5*theta2) +
          2*(x**2 + y**2))
      
      #a = (np.max(z) - z)/(np.max(z) - np.min(z))
      z_list.append(z)
    reg_z_list = []
    for j in z_list:
      mx = np.max(z_list)
      mn = np.min(z_list)
      if j == np.min(z_list):
        reg_z_list.append(0)
      else:
        a = (mx - j)/(mx - mn)
        reg_z_list.append(a)
    return np.asarray(reg_z_list)


def Plotly_data(points, complex_s):
    #points are the given data points, 
    #complex_s is the list of indices in the array of points defining 2-simplexes(triangles) 
    #in the simplicial complex to be plotted
    X=[]
    Y=[]
    for s in complex_s:
        X+=[points[s[k]][0] for k in [0,1,2,0]]+[None]
        Y+=[points[s[k]][1] for k in [0,1,2,0]]+[None]
    return X,Y

colors=['#C0223B', '#404ca0', 'rgba(173,216,230, 0.5)']
def make_trace(x, y,  point_color=colors[0], line_color=colors[1]):# define the trace
                                                                   #for an alpha complex
    return Scatter(mode='markers+lines', #set vertices and 
                                         #edges of the alpha-complex
                   name='',
                   x=x,
                   y=y,
                   marker=Marker(size=6.5, color=point_color),
                   line=Line(width=1.25, color=line_color),

                  )


levels = np.arange(0., 1., 0.025)
cmap = cm.get_cmap(name='Blues', lut=None)

def plot_samples(num):
  samples_hundred = BM_sampling_method(num)
  tri = Triangulation(samples_hundred[0], samples_hundred[1])
  random_gen = np.random.mtrand.RandomState(seed=127260)
  init_mask_frac = 0.0
  min_circle_ratio = .01 
  subdiv = 3

  ntri = tri.triangles.shape[0]

  print 'hi'
  mask_init = np.zeros(ntri, dtype=np.bool)
  masked_tri = random_gen.randint(0, ntri, int(ntri*init_mask_frac))
  mask_init[masked_tri] = True
  tri.set_mask(mask_init)
  print 'hey'

  z_exp = experiment_res(tri.x, tri.y)
  print z_exp

  mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
  tri.set_mask(mask)

# refining the data
  refiner = UniformTriRefiner(tri)
  tri_refi, z_test_refi = refiner.refine_field(z_exp, subdiv=subdiv)

# analytical 'results' for comparison
  z_expected = experiment_res(tri_refi.x, tri_refi.y)

  plt.tricontour(tri_refi, z_expected, levels=levels, cmap=cmap,
                   linestyles='--')


  plt.show()
  x, y = np.mgrid[-1:1:0.001, -1:1:0.001]
  pos = np.empty(x.shape + (2,))
  pos[:, :, 0] = x; pos[:, :, 1] = y
  rv = multivariate_normal([0, 0], [[1.0, 0.0], [0.0, 1.0]])
  plt.contour(x, y, rv.pdf(pos))
  #plt.show()

#plot_samples(100)
plot_samples(500)



