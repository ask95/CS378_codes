#referred to Matthias Baas' implementation

#from hammersley import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay, Voronoi
from plotly.graph_objs import *
import plotly.offline as ploo

def planeHammersley(s,n):

    samples = np.zeros((s, n))

    for k in range(n):
        u = 0
        p = 0.5
        base = s
        kk = k
        while kk > 0:
            if kk%base:
                u += p
            p *= 0.5
            kk = int(kk/base)
        v = (k+0.5)/n
        #yield (u, v)
        samples[0][k] = float(u)
        samples[1][k] = float(v)
    return samples

#UNIT SQUARE IS SAMPLED 500 points
samples = planeHammersley(2,500)

sorted(samples, key=lambda x: x[0])
#print samples[:, :10]
plt.scatter(samples[0], samples[1])
plt.show()

#tri = Delaunay(samples)
#X, Y = Plotly_data(samples, tri.simplices)

max_discr = -100
total_num_points = len(samples[0])

for p_ix in range(total_num_points):
	incl_points = 0
	for inc_p_idx in range(p_ix):
		if samples[1][inc_p_idx] <= samples[1][p_ix]:
			incl_points += 1

	#print incl_points
	area = samples[0][p_ix]*samples[1][p_ix]
	#print area
	dpr = (incl_points*1.0000/total_num_points) - area
	#print dpr

	if dpr > max_discr:
		max_discr = dpr

print max_discr






