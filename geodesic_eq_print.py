import sympy
from metric import *

schwarzschild, variables = getMetric("schwarzschild")
kerr, variables = getMetric("kerr")

def print_geodesic_rhs(metric, latex = False):
    """
    Printing geodesic equation right hand sides. If latex = False they are printed
    in Python syntax, whereas with latex = True they are printed in LaTeX syntax
    """
    for i in range(metric.dim):
        rhs = sympy.latex(metric.geodesic_rhs[i]) if latex else metric.geodesic_rhs[i]
        print(rhs, "\n")

    print("\n")

# With Python syntax
print_geodesic_rhs(schwarzschild)
print_geodesic_rhs(kerr)

# With LaTeX syntax
print_geodesic_rhs(schwarzschild, latex = True)
print_geodesic_rhs(kerr, latex = True)
