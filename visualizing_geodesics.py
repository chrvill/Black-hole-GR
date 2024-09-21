import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kerr_geodesic import *
from color_calculator import *

def animate_geodesics(a, n_geodesics, n_frames, direction = "prograde"):
    kerr = Kerr(parameters = {"a": a})

    t0 = 0
    r0 = 10
    theta0 = np.pi/2
    phi0 = 0

    n = 10
    wavelengths = np.linspace(400, 650, n_geodesics)*1e-9

    p1 = -1
    p2 = 0

    colorhandler = ColorHandler()

    fig = plt.figure()
    ax = plt.axes(xlim = (-10, 10), ylim = (-10, 10))

    lines = []
    x_arrays = []
    y_arrays = []

    n_frames = 500

    sign = 1 if direction == "prograde" else -1

    T = np.linspace(0, 50, n_frames)

    for i in range(n_geodesics):
        p3 = sign*0.01*i
        p0 = kerr.compute_p0(r0, theta0, p1, p2, p3, mu = 0)

        y0 = [t0, r0, theta0, phi0, p0, p1, p2, p3]

        # Solve the equations of motion for a specified number of steps with a specified initial step size
        tau, sol = kerr.solve(y0, 1000, 1e-5)

        t, r, theta, phi = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

        x = np.sqrt(r**2 + kerr.a**2)*np.cos(phi)*np.sin(theta)
        y = np.sqrt(r**2 + kerr.a**2)*np.sin(phi)*np.sin(theta)

        x_samples = np.interp(T, t, x)
        y_samples = np.interp(T, t, y)

        x_arrays.append(x_samples)
        y_arrays.append(y_samples)

        rgb = colorhandler.RGB_monochromatic(wavelengths[i])
        #ax.plot(x, y, color = rgb)
        line, = ax.plot([], [], color = rgb)
        lines.append(line)
    
    x_ergo, y_ergo = compute_r_ergosphere(kerr)
    EH = plt.Circle((0, 0), np.sqrt(kerr.r_EH**2 + kerr.a**2), color = "k")
    ax.add_patch(EH)

    ax.plot(x_ergo, y_ergo, "r--", label = "Ergosphere")
    ax.set_xlabel(r"$x/M$")
    ax.set_ylabel(r"$y/M$")
    ax.legend()
    ax.set_aspect("equal")

    anim = FuncAnimation(fig, lambda i: animate(i, x_arrays, y_arrays, lines), frames = n_frames, interval = 1)
    fig.tight_layout()
    anim.save(f"images/geodesics_{direction}.mp4", fps = 120, dpi = 500)
    fig.savefig(f"images/geodesics_{direction}.pdf", bbox_inches = "tight")

def animate(i, x_arrays, y_arrays, lines):
    for j in range(len(lines)):
        x = x_arrays[j][:i]
        y = y_arrays[j][:i]
        lines[j].set_data(x, y)

    return lines

def compute_r_ergosphere(kerr):
    theta = np.linspace(0, 2*np.pi, 100)

    x_ergo = np.sqrt(2**2 + kerr.a**2)*np.cos(theta)
    y_ergo = np.sqrt(2**2 + kerr.a**2)*np.sin(theta)

    return x_ergo, y_ergo

animate_geodesics(1, 10, 500, "retrograde")
animate_geodesics(1, 10, 500, "prograde")