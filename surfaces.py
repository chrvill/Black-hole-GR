import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

a = np.concatenate((np.linspace(0, 1, 500), np.ones(100)))

theta = np.linspace(0, 2*np.pi, 100)

fig = plt.figure()
ax = plt.axes(xlim = (-2.5, 2.5), ylim = (-2.5, 3.5))

lines = []
styles = ["r-", "k-", "k--", "r--"]
labels = ["Outer ergosphere", "Outer EH", "Inner EH", "Inner ergosphere"]

lines = []
for i in range(4):
    line, = ax.plot([], [], styles[i], label = labels[i])
    lines.append(line)

a_text = ax.text(0.02, 0.95, "", transform = ax.transAxes)

def animate(i):
    r_EH_plus = 1 + np.sqrt(1 - a[i]**2)
    r_EH_min  = 1 - np.sqrt(1 - a[i]**2)

    r_ergo_plus = 1 + np.sqrt(1 - a[i]**2*np.cos(theta)**2)
    r_ergo_min  = 1 - np.sqrt(1 - a[i]**2*np.cos(theta)**2)

    r = [r_ergo_plus, r_EH_plus, r_EH_min, r_ergo_min]
        
    for j in range(len(lines)):
        x = np.sqrt(r[j]**2 + a[i]**2)*np.sin(theta)
        z = r[j]*np.cos(theta) 
        
        lines[j].set_data(x, z)
        
    a_text.set_text('a = {:.2f}'.format(a[i]))

    return lines + [a_text]

anim = FuncAnimation(fig, animate, frames = len(a), interval = 0.5, blit = True)
fig.tight_layout()
ax.set_aspect("equal")
ax.legend()
anim.save("images/ergosphere_and_EH.mp4", fps = 120, dpi = 500)