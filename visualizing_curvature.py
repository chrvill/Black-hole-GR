import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

from kerr_geodesic import *

a = np.linspace(0, 1, 1000)

n_geodesics = 12 

delta_phi = 2*np.pi/n_geodesics

x = []

r0 = 10
theta0 = np.pi/2
pr = -1

fig = plt.figure()
ax = plt.axes(xlim = (-11, 11), ylim = (-11, 11))

lines = []
x_arrays = []
y_arrays = []

lines = []

x_tot = []
y_tot = []

for j in range(n_geodesics):
    line, = ax.plot([], [], "b-")
    lines.append(line)

for i in range(len(a)):
    kerr = KerrMetric(a = a[i])
    gtph = kerr.g_tph(r0, theta0)
    gphph = kerr.g_phph(r0, theta0)
    grr = kerr.g_rr(r0, theta0)
    gtt = kerr.g_tt(r0, theta0)
        
    if a[i] == 0:
        p3 = 0
        p0 = -grr/gtt*pr
    else:
        p3 = np.sqrt(-grr*(gtt*gphph**2/gtph**2 - gphph)**(-1))*pr 
        p0 = -gphph/gtph*p3 
            
    #y0 = [0, r0, theta0, delta_phi*i, p0, 0, 0, p3]
    y0 = [0, r0, theta0, 0, p0, 0, 0, p3]
        
    tau, solution = solve(y0, 1000, 1e-5, kerr)
        
    r, phi = solution[:, 1], solution[:, 3]
        
    #x_arrays.append(x)
    #y_arrays.append(y)
    
    #x_tot.append(x_arrays)
    #y_tot.append(y_arrays)
    
    x_arrays = []
    y_arrays = []
    
    for j in range(n_geodesics):
        x = np.sqrt(r**2 + kerr.a**2)*np.cos(phi + delta_phi*j)
        y = np.sqrt(r**2 + kerr.a**2)*np.sin(phi + delta_phi*j)
        
        x_arrays.append(x)
        y_arrays.append(y)
    
    x_tot.append(x_arrays)
    y_tot.append(y_arrays)
        
EH = None
a_text = ax.text(0.02, 0.95, "", transform = ax.transAxes)

def animate(i):
    global EH 
    if EH:
        EH.remove()
    for j in range(n_geodesics):
        lines[j].set_data(x_tot[i][j], y_tot[i][j])
        
    r_EH = 1 + np.sqrt(1 - a[i]**2)
    radius = np.sqrt(r_EH**2 + a[i]**2)
    EH = plt.Circle((0, 0), radius, color = "k")
    ax.add_patch(EH)
    
    a_text.set_text('a = {:.2f}'.format(a[i]))
    
    return lines

x_EH, y_EH, x_ergo, y_ergo = compute_EH_and_ergosphere(kerr)
#EH = plt.Circle((0, 0), np.sqrt(kerr.r_EH**2 + kerr.a**2), color = "k")
#ax.add_patch(EH)

ax.plot(x_ergo, y_ergo, "r--", label = "Ergosphere")
thetas = np.linspace(0, 2*np.pi, 100)
r = [10, 8, 6, 4]

for r_i in r:
    ax.plot(r_i*np.cos(thetas), r_i*np.sin(thetas), "b-")


ax.set_xlabel(r"$x/M$")
ax.set_ylabel(r"$y/M$")
ax.legend()
ax.set_aspect("equal")

anim = FuncAnimation(fig, animate, frames = len(a), interval = 1, blit = True)
fig.tight_layout()
anim.save("images/curvature.mp4", fps = 120, dpi = 500)