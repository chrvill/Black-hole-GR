import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton
from geodesic_solver import GeodesicSolver 

class SchwarzschildDeSitter(GeodesicSolver):
    def __init__(self, parameters):
        GeodesicSolver.__init__(self, parameters)
        self.H = parameters["H"]
        
        self.r_EH = newton(self.g_tt, 2, fprime = self.g_tt_derivative)
        self.r_cosm = newton(self.g_tt, 1e5, fprime = self.g_tt_derivative)
        print(self.r_EH)
        print(self.r_cosm)
        
    def g_tt_derivative(self, r, theta = np.pi/2): 
        return -(2/r**2 - 2*self.H**2*r)
    
    def g_tt(self, r, theta = np.pi/2):
        return -(1 - 2/r - self.H**2*r**2)

    def g_rr(self, r, theta = np.pi/2):
        return 1/(1 - 2/r - self.H**2*r**2)
    
    def g_thth(self, r, theta = np.pi/2):
        return r**2
    
    def g_phph(self, r, theta):
        return r**2*np.sin(theta)**2
        
    def geodesic_eq_t(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y 
        H = self.H 
        return -1.0*p0*p1*r*(2*H**2*r - 2/r**2)/(H**2*r**3 - r + 2) 
    
    def geodesic_eq_r(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y 
        H = self.H 
        return -p0**2*(-2*H**2*r + 2/r**2)*(-0.5*H**2*r**2 + 0.5 - 1.0/r) - p1**2*(2*H**2*r - 2/r**2)*(-0.5*H**2*r**2 + 0.5 - 1.0/r)/(-H**2*r**2 + 1 - 2/r)**2 + 2*p2**2*r*(-0.5*H**2*r**2 + 0.5 - 1.0/r) + 2*p3**2*r*(-0.5*H**2*r**2 + 0.5 - 1.0/r)*np.sin(theta)**2 
    
    def geodesic_eq_theta(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y 
        H = self.H 
        return -2.0*p1*p2/r + 1.0*p3**2*np.sin(theta)*np.cos(theta)
    
    def geodesic_eq_phi(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y 
        H = self.H     
        return -2.0*p1*p3/r - 2.0*p2*p3*np.cos(theta)/np.sin(theta)
    
    def plotGeodesic(self, geodesic):
        r = geodesic[:, 1]
        theta = geodesic[:, 2]
        phi = geodesic[:, 3]

        x = r*np.cos(phi)*np.sin(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(theta)

        fig, ax = plt.subplots()
        ax.plot(x, y, "b-", label = "Geodesic")
        ax.plot(x[0], y[0], "ro", label = "Start")
        
        EH = plt.Circle((0, 0), self.r_EH, color = "k")
        ax.add_patch(EH)

        ax.set_xlabel(r"$x/M$")
        ax.set_ylabel(r"$y/M$")

        ax.legend()
        ax.set_aspect("equal")
        fig.savefig("images/SdS_geodesic.pdf", bbox_inches = "tight")
    
if __name__ == "__main__":
    LambdaMax = 1/9
    Hmax = np.sqrt(LambdaMax/3)
    H = Hmax*0.9
    print(f"H = {H}")      
    
    SdS = SchwarzschildDeSitter(parameters = {"H": H})
    
    t0 = 0
    r0 = 4
    theta0 = np.pi/2
    phi0 = 0
    
    p1 = 0
    p2 = 0
    p3 = 0.0
    p0 = SdS.compute_p0(r0, theta0, p1, p2, p3, mu = -1)
    
    y0 = np.array([t0, r0, theta0, phi0, p0, p1, p2, p3])
    
    tau, geodesic = SdS.solve(y0, 3000, 1e-7)
    
    SdS.plotGeodesic(geodesic)