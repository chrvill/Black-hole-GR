import sympy 
from metric import *

class MetricWithEM(Metric):
    def __init__(self, g, coordinates, four_potential):
        self.four_potential = four_potential # The four-potential A_{\mu}
        
        Metric.__init__(self, g, coordinates)
        
    def F_covariant(self, mu, nu):
        """
        Computing the electromagnetic field tensor F_{\mu \nu} = \partial_{\mu} A_{\nu} - \partial_{\nu} A_{\mu}
        """
        return sympy.diff(self.four_potential[nu], self.coordinates[mu]) - sympy.diff(self.four_potential[mu], self.coordinates[nu])
        
    def F_contravariant(self, mu, nu):
        """
        The contraviariant components F^{mu nu} = g^{\mu rho} g^{\nu sigma} F_{rho sigma} of the electromagnetic field tensor
        """
        F = sympy.Float(0)
        
        for rho in range(self.dim):
            for sigma in range(self.dim):
                F += self.g_inv[mu, rho]*self.g_inv[nu, sigma]*self.F_covariant(rho, sigma)
                
        return F
    
    def compute_geodesic_rhs(self, mu):
        """
        Computing the mu-th component of the geodeisc equation, where we also get a contribution from the electromagnetic field tensor
        q is the charge of the particle we're studying
        """
        result = sympy.Float(0)
        
        for rho in range(self.dim):
            for sigma in range(self.dim):
                result -= self.compute_christoffel(mu, rho, sigma)*self.momentum[rho]*self.momentum[sigma]

                result += q*self.g[sigma, rho]*self.F_contravariant(mu, rho)*self.momentum[sigma]
        
        return result

# Q is the charge of the black hole
t, r, theta, phi, a, Q, q = sympy.symbols("t, r, theta, phi, a, Q, q")

coordinates = [t, r, theta, phi]

# Setting up the components of the metric tensor and four-potential
rQ = Q/sympy.sqrt(4*sympy.pi)

rho2 = r**2 + a**2*sympy.cos(theta)**2
delta = r**2 - 2*r + a**2 + rQ**2

A_mu = [r*rQ/rho2, 0, 0, -a*r*rQ*sympy.sin(theta)**2/rho2]

g = sympy.Matrix([[-delta/rho2 + a**2*sympy.sin(theta)**2/rho2, 0, 0, delta/rho2*a*sympy.sin(theta)**2 - sympy.sin(theta)**2/rho2*a*(r**2 + a**2)],
                  [0, rho2/delta, 0, 0],
                  [0, 0, rho2, 0],
                  [delta/rho2*a*sympy.sin(theta)**2 - sympy.sin(theta)**2/rho2*a*(r**2 + a**2), 0, 0, -delta/rho2*a**2*sympy.sin(theta)**4 + sympy.sin(theta)**2/rho2*(r**2 + a**2)**2]])

kerr_newman = MetricWithEM(g, coordinates, A_mu)

for i in range(kerr_newman.dim):
    rhs = kerr_newman.geodesic_rhs[i]
    print(rhs, "\n\n")