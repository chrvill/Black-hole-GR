import sympy

# Pretty printing
sympy.init_printing()

class Metric:
    """
    Input arguments:
        g:
            sympy.Matrix
            Contains the covariant components of the metric tensor (g_(mu nu))

        coordinates:
            list/array of sympy.Symbol
            Contains the coordinates in which the metric tensor is expressed

    The class has methods for:
        - differentiating the metric tensor with respect to given coordinate
        - computing Christoffel symbols of the metric
        - computing the right-hand-side of a given component of the geodesic equation
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

def getMetric(name):
    """
    Takes as input argument the name of a metric, either "kerr" or "schwarzschild"
    and returns a Metric-object containing the corresponding metric. Also returns
    a list containing all the variables entering the metric, as sympy.symbols.
    """
    t, r, theta, phi, M, a, v = sympy.symbols("t, r, theta, phi, M, a, v")

    coordinates = [t, r, theta, phi]

    sigma = r**2 + a**2*sympy.cos(theta)**2
    delta = r**2 - 2*r + a**2

    # Kerr metric
    g = sympy.Matrix([[-(1 - 2*r/sigma), 0, 0, -2*a*r/sigma*sympy.sin(theta)**2],
                      [0, sigma/delta, 0, 0], [0, 0, sigma, 0],
                      [-2*a*r/sigma*sympy.sin(theta)**2, 0, 0,
                      (r**2 + a**2 + 2*r*a**2/sigma*sympy.sin(theta)**2)*sympy.sin(theta)**2]])

    kerr = Metric(g, coordinates)

    # Schwarzschild metric
    g = sympy.Matrix([[-(1 - 2*M/r), 0, 0, 0],
                      [0, (1 - 2*M/r)**(-1), 0, 0],
                      [0, 0, r**2, 0],
                      [0, 0, 0, r**2*sympy.sin(theta)**2]])

    schwarzschild = Metric(g, coordinates)

    g = sympy.Matrix([[-(1 - 2/r), 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, r**2, 0],
                      [0, 0, 0, r**2*sympy.sin(theta)**2]])

    coordinates = [v, r, theta, phi]
    schwarzschild_edfink = Metric(g, coordinates)

    metrics = {"kerr": [kerr, [t, r, theta, phi, M, a]], "schwarzschild": [schwarzschild, [t, r, theta, phi, M]], \
               "schwarzschild_edfink": [schwarzschild_edfink, [v, r, theta, phi]]}
    return metrics[name]
