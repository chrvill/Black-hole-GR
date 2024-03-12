import sympy

# Pretty printing
sympy.init_printing()

class Metric:
    """
    Input arguments:
        g:
            sympy.Matrix
            Contains the covariant components of the metric tensor (g_{\mu \nu})

        coordinates:
            list/array of sympy.Symbol
            Contains the coordinates in which the metric tensor is expressed

    The class has methods for:
        - differentiating the metric tensor with respect to given coordinate
        - computing Christoffel symbols of the metric
        - computing the right-hand-side of a given component of the geodesic equation
        - computing Riemann tensor and Ricci tensor and scalar
    """

    def __init__(self, g, coordinates):
        self.g = g                      # The covariant components of the metric tensor
        self.g_inv = self.g.inv()       # The contravariant ---------||----------------
        self.coordinates = coordinates  # The coordinates in which we express the metric

        self.dim = len(self.coordinates)  # Dimension of the spacetime in question

        # Sympy variables representing the components of the four-momentum of a particle
        # p0 is dx^0/dlambda, p1 is dx^1/dlambda etc.
        momentum_string = ", ".join([f"p{i}" for i in range(self.dim)]) 
        self.momentum = sympy.symbols(momentum_string)
        
        # List containing the right-hand-side (rhs) of each component of the geodesic equation
        self.geodesic_rhs = [self.compute_geodesic_rhs(i) for i in range(self.dim)]

    def deriv(self, mu, nu, rho):
        """
        Differentiates the metric tensor component g_(mu nu) with respect to coordinate rho (corresponding to self.coordinates[rho])
        """
        return sympy.diff(self.g[mu, nu], self.coordinates[rho])

    def compute_christoffel(self, mu, rho, sigma):
        """
        Computes the Christoffel symbol Gamma^(mu)_(rho sigma) by calling the function self.deriv()
        """
        result = sympy.Float(0)

        for nu in range(self.dim):
            result += 1.0/2.0*self.g_inv[mu, nu]*(self.deriv(nu, rho, sigma) + self.deriv(nu, sigma, rho) - self.deriv(rho, sigma, nu))

        return result

    def compute_geodesic_rhs(self, mu):
        """
        Computes the mu-th component of the geodesic equation, which comes down to calculating the Christoffel symbols
        """
        result = sympy.Float(0)

        for rho in range(self.dim):
            for sigma in range(self.dim):
                result -= self.compute_christoffel(mu, rho, sigma)*self.momentum[rho]*self.momentum[sigma]

        return result
    
    def compute_riemann_tensor(self, rho, sigma, mu, nu):
        """
        Computes the component R^(rho)_(sigma mu nu) of the Riemann tensor
        """
        result = sympy.diff(self.compute_christoffel(rho, nu, sigma), self.coordinates[mu]) \
               - sympy.diff(self.compute_christoffel(rho, mu, sigma), self.coordinates[nu])
        
        for lambda_ in range(self.dim):
            result += self.compute_christoffel(rho, mu, lambda_)*self.compute_christoffel(lambda_, nu, sigma)
            result -= self.compute_christoffel(rho, nu, lambda_)*self.compute_christoffel(lambda_, mu, sigma)
        
        return result
    
    def compute_ricci_tensor(self, mu, nu):
        """
        Computes the component R_(mu nu) of the Ricci tensor R^lambda_(mu lambda nu)
        """
        result = sympy.Float(0)
        
        for lambda_ in range(self.dim):
            result += self.compute_riemann_tensor(lambda_, mu, lambda_, nu)
            
        return result

    def compute_ricci_scalar(self):
        """ 
        Compute the Ricci scalar R = g^(mu nu)R_(mu nu)
        """
        result = sympy.Float(0)
        
        for mu in range(self.dim):
            for nu in range(self.dim):
                result += self.g_inv[mu, nu]*self.compute_ricci_tensor(mu, nu)
                
        return result

def getKerr():
    """
    Returns a dictionary containing
    - the Kerr metric
    - The Boyer Lindquist coordinates in which it is expressed
    - The angular momentum parameter a
    """
    t, r, theta, phi, a = sympy.symbols("t, r, theta, phi, a")
    
    coordinates = [t, r, theta, phi]
    
    sigma = r**2 + a**2*sympy.cos(theta)**2
    delta = r**2 - 2*r + a**2

    # Kerr metric
    g = sympy.Matrix([[-(1 - 2*r/sigma), 0, 0, -2*a*r/sigma*sympy.sin(theta)**2],
                      [0, sigma/delta, 0, 0], [0, 0, sigma, 0],
                      [-2*a*r/sigma*sympy.sin(theta)**2, 0, 0,
                      (r**2 + a**2 + 2*r*a**2/sigma*sympy.sin(theta)**2)*sympy.sin(theta)**2]])

    kerr = Metric(g, coordinates)
    
    data = {"metric": kerr, "coordinates": coordinates, "parameters": {"a": a}}
    
    return data

def getSchwarzschild():
    """
    Returns a dictionary containing
    - The Schwarzschild metric
    - The Schwarzschild coordinates in which it is expressed
    """
    t, r, theta, phi = sympy.symbols("t, r, theta, phi")
    
    coordinates = [t, r, theta, phi]
    
    g = sympy.Matrix([[-(1 - 2/r), 0, 0, 0],
                      [0, (1 - 2/r)**(-1), 0, 0],
                      [0, 0, r**2, 0],
                      [0, 0, 0, r**2*sympy.sin(theta)**2]])

    schwarzschild = Metric(g, coordinates)
    
    data = {"metric": schwarzschild, "coordinates": coordinates}
    
    return data

def getFLRW():
    """
    Returns a dictionary containing
    - The FLRW metric
    - The comoving, spherical coordinates in which it is expressed
    - The scale factor a(t) as a sympy.Function object 
    """
    t, r, theta, phi = sympy.symbols("t, r, theta, phi")
    
    # The scale factor a(t)
    a_func = sympy.Function("a")
    a = a_func(t)
    
    coordinates = [t, r, theta, phi]
    
    g = sympy.Matrix([[-1, 0, 0, 0], 
                      [0, a**2, 0, 0],
                      [0, 0, a**2*r**2, 0],
                      [0, 0, 0, a**2*r**2*sympy.sin(theta)**2]])
                     
    flrw = Metric(g, coordinates)
    
    data = {"metric": flrw, "coordinates": coordinates, "parameters": {"a": a}}    

    return data

def getAlcubierre():
    """
    Returns a dictionary containing
    - The Alcubierre metric
    - The coordinates in which it is expressed
    - f(r_s)
    - x_s(t) as a sympy.Function object
    """
    t, x, y, z, sigma, R = sympy.symbols("t, x, y, z, sigma, R")
    
    x_s = sympy.Function("x_s")
    x_s = x_s(t)
    v_s = sympy.Derivative(x_s)
    
    r_s = sympy.sqrt((x - x_s)**2 + y**2 + z**2)
    
    f = (sympy.tanh(sigma*(r_s + R)) - sympy.tanh(sigma*(r_s - R)))/(2*sympy.tanh(sigma*R))

    coordinates = [t, x, y, z]
    
    g = sympy.Matrix([[v_s**2*f**2 - 1, -v_s*f, 0, 0], 
                      [-v_s*f, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    alcubierre = Metric(g, coordinates)
    
    data = {"metric": alcubierre, "coordinates": coordinates, "parameters": {"f": f, "x_s": x_s}}
    
    return data

if __name__ == "__main__":
    """
    flrw, coordinates, a = getFLRW()

    t = coordinates[0]

    geodesic_eq = flrw.geodesic_rhs[0]
    print(geodesic_eq.subs(a, sympy.exp(t)).doit())
    """

    data = getAlcubierre()
    alcubierre = data["metric"]
    coordinates = data["coordinates"]
    x_s = data["parameters"][1]

    t = coordinates[0]

    v = sympy.symbols("v")

    for i in range(alcubierre.dim):
        rhs = alcubierre.geodesic_rhs[i]
        print(rhs.subs(x_s, v*t).doit(), "\n\n")
