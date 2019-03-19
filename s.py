import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# plt.figure(figsize=(10,8))
x = np.linspace(0,5,30)
y = 3*x

# plt.subplot(2,1,1)
# plt.plot(x,y, label='1')
# plt.legend(loc="best")

# plt.subplot(2,3,4)
# plt.plot(x,y, label='2')
# plt.legend(loc="best")

# plt.subplot(2,3,5)
# plt.plot(x,y, label='3')
# plt.legend(loc="best")

# plt.subplot(2,3,6)
# plt.plot(x,y, label='4')
# plt.legend(loc="best")

# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
# ax1.plot(x,y)
# ax1.set_title('ax1 title')

# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# ax2.plot(x,y)
# ax2.set_title('ax2 title')

# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# ax3.plot(x,y)
# ax3.set_title('ax3 title')

# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax4.plot(x,y)
# ax4.set_title('ax4 title')

# ax5 = plt.subplot2grid((3, 3), (2, 1))
# ax5.plot(x,y)
# ax5.set_title('ax5 title')
# fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(12,8))
# for i,ax in enumerate(ax.ravel()):
# 	ax.plot(x,y)
# 	ax.set_title('ax%d'%i)
# for axx in ax:
# 	for axxx in axx:
# 		axxx.plot(x,y)
# plt.tight_layout()

fig = plt.figure(num='3d plot',figsize=(10,8))


# ax=Axes3D(fig)
# ax=fig.gca(projection='3d')

# X = np.arange(-4,4,0.05)
# Y = np.arange(-4,4,0.05)
# # X,Y = np.meshgrid(X,Y)
# X,Y=np.mgrid[-4:4:100j,-4:4:100j]
# R = np.sqrt(X**2+Y**2)

# Z=np.sin(R)
# ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=plt.get_cmap('rainbow'))
# ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
# ax.set_ylim(-5, 5)
# ax.set_xlim(-5,5)
# ax.set_zlim(-2,2)
# ax.set_xlabel('x axis')
# ax.set_ylabel('y axis')
# plt.title('3d surface')
# ax.plot(x,y,zdir='x')


# n_radii = 8
# n_angles = 36
 
# radii = np.linspace(0.125, 1.0, n_radii)
# angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
 
# angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
 
# x = np.append(0, (radii*np.cos(angles)).flatten())
# y = np.append(0, (radii*np.sin(angles)).flatten())
 
# z = np.sin(-x*y)
 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
 
# ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)


# a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
#               0.365348418405, 0.439599930621, 0.525083754405,
#               0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)
# plt.imshow(a, interpolation='bicubic', cmap='prism', origin='lower')
# plt.colorbar(shrink=.92)

def f(x,y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)
x,y = np.mgrid[-3:3:256j,-3:3:256j]
z = f(x,y)
plt.contourf(x,y,z,30,alpha=0.75,cmap=plt.cm.autumn)
c=plt.contour(x,y,z,30,color='black',linewidth=10)
plt.clabel(c,fontsize=10,inline=True)
plt.xticks(())
plt.yticks(())
plt.colorbar()

plt.show()