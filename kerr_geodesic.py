import numpy as np 
import matplotlib.pyplot as plt 
from geodesic_solver import GeodesicSolver

class Kerr(GeodesicSolver):
    def __init__(self, parameters):
        GeodesicSolver.__init__(self, parameters)
        self.a = parameters["a"]
        self.r_EH = 1 + np.sqrt(1 - self.a**2)
        
    def g_tt(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2

        return -(1 - 2*r/sigma)

    def g_tph(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2

        return -2*self.a*r/sigma*np.sin(theta)**2

    def g_rr(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2
        delta = r**2 - 2*r + self.a**2

        return sigma/delta

    def g_thth(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2

        return sigma

    def g_phph(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2
        delta = r**2 - 2*r + self.a**2
        Lambda = (r**2 + self.a**2)**2 - self.a**2*delta*np.sin(theta)**2

        return Lambda/sigma*np.sin(theta)**2    
    
    def geodesic_eq_t(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y  
        a = self.a 
        sin = np.sin(theta)
        cos = np.cos(theta)
        
        return -2*p0*p1*(-1.0*a*r*(4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 - 2*a*sin**2/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(-4*r**2/(a**2*cos**2 + r**2)**2 + 2/(a**2*cos**2 + r**2))*(-a**4*cos**2 - a**2*r**2*cos**2 - a**2*r**2 - 2*a**2*r*sin**2 - r**4)/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) - 2*p0*p2*(2.0*a**2*r*(-a**4*cos**2 - a**2*r**2*cos**2 - a**2*r**2 - 2*a**2*r*sin**2 - r**4)*sin*cos/((a**2*cos**2 + r**2)**2*(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) - 1.0*a*r*(-4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 - 4*a*r*sin*cos/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) - 2*p1*p3*(-1.0*a*r*(-4*a**2*r**2*sin**2/(a**2*cos**2 + r**2)**2 + 2*a**2*sin**2/(a**2*cos**2 + r**2) + 2*r)*sin**2/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 - 2*a*sin**2/(a**2*cos**2 + r**2))*(-a**4*cos**2 - a**2*r**2*cos**2 - a**2*r**2 - 2*a**2*r*sin**2 - r**4)/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) - 2*p2*p3*(-1.0*a*r*((4*a**4*r*sin**3*cos/(a**2*cos**2 + r**2)**2 + 4*a**2*r*sin*cos/(a**2*cos**2 + r**2))*sin**2 + 2*(2*a**2*r*sin**2/(a**2*cos**2 + r**2) + a**2 + r**2)*sin*cos)/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(-4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 - 4*a*r*sin*cos/(a**2*cos**2 + r**2))*(-a**4*cos**2 - a**2*r**2*cos**2 - a**2*r**2 - 2*a**2*r*sin**2 - r**4)/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3))
    
    def geodesic_eq_r(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y  
        a = self.a 
        sin = np.sin(theta)
        cos = np.cos(theta)
        
        return 2.0*a**2*p1*p2*sin*cos/(a**2*cos**2 + r**2) + 1.0*r*p2**2*(a**2 + r**2 - 2*r)/(a**2*cos**2 + r**2) - 0.5*p0**2*(4*r**2/(a**2*cos**2 + r**2)**2 - 2/(a**2*cos**2 + r**2))*(a**2 + r**2 - 2*r)/(a**2*cos**2 + r**2) - 1.0*p0*p3*(-4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 + 2*a*sin**2/(a**2*cos**2 + r**2))*(a**2 + r**2 - 2*r)/(a**2*cos**2 + r**2) - 0.5*p1**2*(2*r/(a**2 + r**2 - 2*r) + (2 - 2*r)*(a**2*cos**2 + r**2)/(a**2 + r**2 - 2*r)**2)*(a**2 + r**2 - 2*r)/(a**2*cos**2 + r**2) + 0.5*p3**2*(a**2 + r**2 - 2*r)*(-4*a**2*r**2*sin**2/(a**2*cos**2 + r**2)**2 + 2*a**2*sin**2/(a**2*cos**2 + r**2) + 2*r)*sin**2/(a**2*cos**2 + r**2)
    
    def geodesic_eq_theta(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y  
        a = self.a 
        sin = np.sin(theta)
        cos = np.cos(theta)
        
        return 2.0*a**2*r*p0**2*sin*cos/(a**2*cos**2 + r**2)**3 - 1.0*a**2*p1**2*sin*cos/((a**2*cos**2 + r**2)*(a**2 + r**2 - 2*r)) + 1.0*a**2*p2**2*sin*cos/(a**2*cos**2 + r**2) - 2.0*r*p1*p2/(a**2*cos**2 + r**2) - 1.0*p0*p3*(4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 + 4*a*r*sin*cos/(a**2*cos**2 + r**2))/(a**2*cos**2 + r**2) - 0.5*p3**2*(-(4*a**4*r*sin**3*cos/(a**2*cos**2 + r**2)**2 + 4*a**2*r*sin*cos/(a**2*cos**2 + r**2))*sin**2 - 2*(2*a**2*r*sin**2/(a**2*cos**2 + r**2) + a**2 + r**2)*sin*cos)/(a**2*cos**2 + r**2)
    
    def geodesic_eq_phi(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y  
        a = self.a 
        sin = np.sin(theta)
        cos = np.cos(theta)
        
        return -2*p0*p1*(-1.0*a*r*(-4*r**2/(a**2*cos**2 + r**2)**2 + 2/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 - 2*a*sin**2/(a**2*cos**2 + r**2))*(a**2*cos**2 + r**2 - 2*r)/(a**4*sin**2*cos**2 + a**2*r**2*sin**2*cos**2 + a**2*r**2*sin**2 + 2*a**2*r*sin**4 - 2*a**2*r*sin**2 + r**4*sin**2 - 2*r**3*sin**2)) - 2*p0*p2*(-4.0*a**3*r**2*sin*cos/((a**2*cos**2 + r**2)**2*(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) + 0.5*(-4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 - 4*a*r*sin*cos/(a**2*cos**2 + r**2))*(a**2*cos**2 + r**2 - 2*r)/(a**4*sin**2*cos**2 + a**2*r**2*sin**2*cos**2 + a**2*r**2*sin**2 + 2*a**2*r*sin**4 - 2*a**2*r*sin**2 + r**4*sin**2 - 2*r**3*sin**2)) - 2*p1*p3*(-1.0*a*r*(4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 - 2*a*sin**2/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(a**2*cos**2 + r**2 - 2*r)*(-4*a**2*r**2*sin**2/(a**2*cos**2 + r**2)**2 + 2*a**2*sin**2/(a**2*cos**2 + r**2) + 2*r)*sin**2/(a**4*sin**2*cos**2 + a**2*r**2*sin**2*cos**2 + a**2*r**2*sin**2 + 2*a**2*r*sin**4 - 2*a**2*r*sin**2 + r**4*sin**2 - 2*r**3*sin**2)) - 2*p2*p3*(-1.0*a*r*(-4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 - 4*a*r*sin*cos/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*((4*a**4*r*sin**3*cos/(a**2*cos**2 + r**2)**2 + 4*a**2*r*sin*cos/(a**2*cos**2 + r**2))*sin**2 + 2*(2*a**2*r*sin**2/(a**2*cos**2 + r**2) + a**2 + r**2)*sin*cos)*(a**2*cos**2 + r**2 - 2*r)/(a**4*sin**2*cos**2 + a**2*r**2*sin**2*cos**2 + a**2*r**2*sin**2 + 2*a**2*r*sin**4 - 2*a**2*r*sin**2 + r**4*sin**2 - 2*r**3*sin**2))

    def plotGeodesic(self, geodesic):
        r = geodesic[:, 1]
        theta = geodesic[:, 2]
        phi = geodesic[:, 3]

        x = np.sqrt(r**2 + self.a**2)*np.cos(phi)*np.sin(theta)
        y = np.sqrt(r**2 + self.a**2)*np.sin(phi)*np.sin(theta)
        z = r*np.cos(theta)

        fig, ax = plt.subplots()
        ax.plot(x, y, "b-", label = "Geodesic")
        ax.plot(x[0], y[0], "ro", label = "Start")

        EH = plt.Circle((0, 0), np.sqrt(self.r_EH**2 + self.a**2), color = "k")
        ax.add_patch(EH)

        circle_angles = np.linspace(0, 2*np.pi, 100)
        x_ergo = np.sqrt(2**2 + self.a**2)*np.cos(circle_angles)
        y_ergo = np.sqrt(2**2 + self.a**2)*np.sin(circle_angles)
        ax.plot(x_ergo, y_ergo, "r--", label = "Ergosphere")

        ax.set_xlabel(r"$x/M$")
        ax.set_ylabel(r"$y/M$")

        ax.legend()
        ax.set_aspect("equal")
        fig.savefig("images/kerr_geodesic.pdf", bbox_inches = "tight")

    def ZAMO_four_vel(self, r, theta):
        """
        Calculates and returns the four-velocity of a ZAMO at Boyer-Lindquist coordinates r and theta
        """
        gtt    = self.g_tt(r, theta)
        gphph = self.g_phph(r, theta)
        gtph  = self.g_tph(r, theta)

        u3 = np.sqrt(1/(gphph - gtt*(gphph/gtph)**2))
        u0 = self.compute_p0(r, theta, 0, 0, u3, mu = -1)
        return np.array([u0, 0, 0, u3])

if __name__ == "__main__":
    a = 1 
    kerr = Kerr(parameters = {"a": a})
    
    t0     = 0
    r0     = 10
    theta0 = np.pi/2
    phi0   = 0

    # Components of the initial four-momentum of the particle
    p1 = -1
    p2 = 0
    p3 = 0
    p0 = kerr.compute_p0(r0, theta0, p1, p2, p3, mu = 0)

    # Collecting the variables we will be interested in evolving in time into a single list
    y0 = [t0, r0, theta0, phi0, p0, p1, p2, p3]

    # Solve the equations of motion for a specified number of steps with a specified initial step size
    tau, geodesic = kerr.solve(y0, 200, 1e-3)
    
    kerr.plotGeodesic(geodesic)
    
    r_final = geodesic[-1, 1]
    theta_final = geodesic[-1, 2]
    
    u_em = np.array([kerr.compute_p0(r0, theta0, 0, 0, 0, mu = -1), 0, 0, 0])
    u_obs = kerr.ZAMO_four_vel(r_final, theta_final)

    redshift = kerr.compute_redshift(geodesic, u_obs, u_em)
    print("1 + z = {}".format(redshift))