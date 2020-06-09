import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# input parameter
den = 8880.0
cp = 386.0
cond = 398.0
temp_bc = 100.0
temp_init = 0.0
lx = 1.0
ly = 1.0
nx = 51
ny = 51
tend = 10000.0
dt = 1.0
tout = 50.0
eps = 1.0e-6
itermax = 1000

alpha = cond / (den * cp)
dx = lx / (nx - 1)
dy = ly / (ny - 1)
nt = int(tend / dt)
nout = int(tout / dt)

ckx = alpha * dt / (dx * dx)
cky = alpha * dt / (dy * dy)

#initial condition
temp = np.full((nx, ny), temp_init) 
time = 0.0

temp_old = np.zeros((nx, ny))

# Boundary condition 
for i in range(nx):
    temp[i][0] = temp[i][1]
    temp[i][ny-1] = temp[i][ny-2]

for j in range(ny):
    temp[0][j] = temp[1][j]
    temp[nx-1][j] = temp[nx-2][j]

for i in range(nx):
    if i * dx <= 0.2:
        temp[i][0] = temp_bc

# graph data array
ims = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

gx = np.zeros(nx)
for i in range(nx):
    gx[i] = i * dx

gy = np.zeros(ny)
for j in range(ny):
    gy[j] = j * dy

X, Y = np.meshgrid(gx, gy)

# time loop
for n in range(1, nt+1):
    # full-implicit with Gauss-Seidel
    for i in range(nx):
        for j in range(ny):
            temp_old[i][j] = temp[i][j]

    for iter in range(itermax):
        resd = 0.0
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                tp = temp[i][j]
                temp[i][j] = 1.0 / (1.0 + 2.0 * ckx + 2.0 * cky) \
                    * (temp_old[i][j] + ckx * temp[i+1][j] + ckx * temp[i-1][j] \
                    + cky * temp[i][j+1] + cky * temp[i][j-1])
                resd += abs(temp[i][j] - tp)

        if resd <= eps:
            break

    # Boundary condition 
    for i in range(nx):
        temp[i][0] = temp[i][1]
        temp[i][ny-1] = temp[i][ny-2]

    for j in range(ny):
        temp[0][j] = temp[1][j]
        temp[nx-1][j] = temp[nx-2][j]

    for i in range(nx):
        if i * dx <= 0.2:
            temp[i][0] = temp_bc

    time += dt

    if n % nout == 0:   
        print('n: {0:7d}, time: {1:8.1f}, temp: {2:10.6f}, iter: {3:4d}'.format(n, time, temp[nx-1][ny-1], iter))
        im_contour = ax.pcolormesh(X, Y, temp.T, cmap='jet', vmin=0.0, vmax=100.0, shading='gouraud')
        im_time = ax.text(0, 1.05, 'Time = {0:8.1f} [s]'.format(time))
        ims.append([im_contour] + [im_time])

# graph plot
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
fig.colorbar(im_contour)
anm = animation.ArtistAnimation(fig, ims, interval=50)
anm.save('animation.gif', writer='pillow')
plt.show()
