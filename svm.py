from cvxopt.solvers import qp
from cvxopt.base import matrix
from pprint import pprint
import numpy , pylab , random , math
import numpy.random as r
np = numpy
pl = pylab

# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

def RBF(x,y,sigma = 0.05):
    return np.exp(-(x-y).dot(x-y)/2/sigma**2)

def linear(x,y):
    return x.dot(y) + 1
            
def polynomial(x,y,p=3):
    return (x.dot(y) +1)**p

def indicator(x,data, t, alpha, kernel = linear):
    return np.sum([alpha[i]*t[i]*kernel(x,data[:,i]) for i in range(t.size)])
    
def genP(data,t,kernel = linear):
    return np.array([[t[i]*t[j]*kernel(data[:,i],data[:,j]) 
            for i in range(t.size)]
            for j in range(t.size)])

N = 20

classA = np.zeros((2,10))
classB = np.zeros((2,10))

classA[0,:5] = r.randn(1,5) -1.5
classA[1,:5] = r.randn(1,5) +0.5
classA[0,5:] = r.randn(1,5) +1.5
classA[1,5:] = r.randn(1,5) +0.5

classB[0,:] = r.randn(1,10)/2
classB[1,:] = r.randn(1,10)/2 + 0.5

data = np.hstack((classA,classB))
t = np.ones(N)
t[N//2:] = -np.ones(N//2)
shuffled = np.vstack((data,t))
np.random.shuffle(shuffled.T)
data = shuffled[:2,:]
t = shuffled[2,:]

kernel = polynomial

P = genP(data,t,kernel)
q = -np.ones(N)
h = np.zeros(N)
G = -np.identity(N)

r = qp ( matrix(P) , matrix( q ), matrix(G) , matrix( h ) )

alpha = list(r['x'])
print("alpha:")
pprint(alpha)

pl.hold(1)
pl.plot(classA[0],classA[1],'bo')
pl.plot(classB[0],classB[1],'ro')

xr = np.arange(-4,4,0.05)
yr = np.arange(-4,4,0.05)

grid = matrix([[indicator(np.array((x,y)),data,t,alpha, kernel) for y in yr] for x in xr])


pl.contour(xr,yr,grid,(-1.0,0.0,1.0),
                colors = ('red','black','blue'),
                linewidths = (1,3,1))

pl.show()
