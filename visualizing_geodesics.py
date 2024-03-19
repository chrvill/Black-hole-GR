import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kerr_geodesic import *
from color_calculator import *

a = 1
kerr = KerrMetric(a = a)

t0 = 0
r0 = 10
theta0 = np.pi/2
phi0 = 0

n = 10
wavelengths = np.linspace(400, 650, n)*1e-9

p1 = -1
p2 = 0

colorhandler = ColorHandler()

fig = plt.figure()
ax = plt.axes(xlim = (-10, 10), ylim = (-10, 10))

lines = []
x_arrays = []
y_arrays = []

n_frames = 500

T = np.linspace(0, 50, n_frames)

for i in range(n):
    p3 = -0.01*i
    p0 = kerr.compute_p0(r0, theta0, p1, p2, p3, mu = 0)

    y0 = [t0, r0, theta0, phi0, p0, p1, p2, p3]

    # Solve the equations of motion for a specified number of steps with a specified initial step size
    tau, sol = solve(y0, 1000, 1e-5, kerr)

    t, r, theta, phi = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

    x = np.sqrt(r**2 + kerr.a**2)*np.cos(phi)*np.sin(theta)
    y = np.sqrt(r**2 + kerr.a**2)*np.sin(phi)*np.sin(theta)

    #x_samples = np.interp(np.linspace(0, np.max(tau), n_frames), tau, x)
    #y_samples = np.interp(np.linspace(0, np.max(tau), n_frames), tau, y)
    x_samples = np.interp(T, t, x)
    y_samples = np.interp(T, t, y)

    x_arrays.append(x_samples)
    y_arrays.append(y_samples)

    rgb = colorhandler.RGB_monochromatic(wavelengths[i])
    #ax.plot(x, y, color = rgb)
    line, = ax.plot([], [], color = rgb)
    lines.append(line)

def animate(i):
    for j in range(n):
        x = x_arrays[j][:i]
        y = y_arrays[j][:i]
        lines[j].set_data(x, y)

    return lines

x_EH, y_EH, x_ergo, y_ergo = compute_EH_and_ergosphere(kerr)
EH = plt.Circle((0, 0), np.sqrt(kerr.r_EH**2 + kerr.a**2), color = "k")
ax.add_patch(EH)

ax.plot(x_ergo, y_ergo, "r--", label = "Ergosphere")
ax.set_xlabel(r"$x/M$")
ax.set_ylabel(r"$y/M$")
ax.legend()
ax.set_aspect("equal")

anim = FuncAnimation(fig, animate, frames = n_frames, interval = 1)
fig.tight_layout()
anim.save("images/geodesics_retrograde.mp4", fps = 120, dpi = 500)
fig.savefig("images/geodesics_retrograde.pdf", bbox_inches = "tight")