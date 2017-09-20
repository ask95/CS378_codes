import numpy as np
import plotly.plotly as py
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, Voronoi
from plotly.graph_objs import *
import plotly.offline as ploo
from statistics import mean, pvariance

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

def experiment_res(x, y):
    """ An analytic function representing experiment results """
    x = list(2.*np.asarray(x))
    r1 = np.sqrt((0.5 - x)**2 + (0.5 - y)**2)
    theta1 = np.arctan2(0.5 - x, 0.5 - y)
    r2 = np.sqrt((-x - 0.2)**2 + (-y - 0.2)**2)
    theta2 = np.arctan2(-x - 0.2, -y - 0.2)
    z = (4*(np.exp((r1/10)**2) - 1)*30. * np.cos(3*theta1) +
         (np.exp((r2/10)**2) - 1)*30. * np.cos(5*theta2) +
         2*(x**2 + y**2))
    return (np.max(z) - z)/(np.max(z) - np.min(z))


def plot_samples(num):
    samples_hundred = BM_sampling_method(num)

    print samples_hundred.T.shape

    #plt.scatter(samples_hundred[0], samples_hundred[1])
    #plt.show()
    m = mean(samples_hundred[0])
    m2 = mean(samples_hundred[1])
    print "mean of" + str(num), m, m2
    print "variance of" + str(num), pvariance(samples_hundred[0], m), pvariance(samples_hundred[1], m2)  

    tri = Delaunay(samples_hundred.T)
    X, Y = Plotly_data(samples_hundred.T, tri.simplices)
    #z_expected = experiment_res(X, Y)
    #plt.contour(X,Y,z)
      #plt.show()
      ##print X, Y
    # data = [make_trace(X, Y)]
    # vor = Voronoi(samples_hundred.T)
    # print "Hey---", len(vor.ridge_vertices)

    # points = samples_hundred.T

    # plt.plot(points[:,0], points[:,1], 'o')
    # plt.plot(vor.vertices[:,0], vor.vertices[:,1], '*')
    # plt.xlim(-1, 3); plt.ylim(-1, 3)

    # for simplex in vor.ridge_vertices:
    #   simplex = np.asarray(simplex)
    #   if np.all(simplex >= 0):
    #       plt.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-')

    # center = points.mean(axis=0)
    
    # for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    #   simplex = np.asarray(simplex)
    #   if np.any(simplex < 0):
    #       i = simplex[simplex >= 0][0] # finite end Voronoi vertex
    #       t = points[pointidx[1]] - points[pointidx[0]] # tangent
    #       t /= np.linalg.norm(t)
    #       n = np.array([-t[1], t[0]]) # normal
    #       midpoint = points[pointidx].mean(axis=0)
    #       far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
    #       plt.plot([vor.vertices[i,0], far_point[0]],
    #                [vor.vertices[i,1], far_point[1]], 'k--')

    # plt.show()

'''
    x, y = np.mgrid[-1:1:0.001, -1:1:0.001]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    rv = multivariate_normal([0, 0], [[1.0, 0.0], [0.0, 1.0]])
    plt.contour(x, y, rv.pdf(pos))
    #plt.show()
'''
plot_samples(1000)




