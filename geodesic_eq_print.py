import sympy
from metric import *

def print_geodesic_rhs(metric, latex = False):
    """
    Printing geodesic equation right hand sides. If latex = False they are printed
    in Python syntax, whereas with latex = True they are printed in LaTeX syntax
    """
    for i in range(metric.dim):
        rhs = sympy.latex(metric.geodesic_rhs[i]) if latex else metric.geodesic_rhs[i]
        print(rhs, "\n")

    print("\n")

### For Schwarzschild 
data_schwarzschild = getSchwarzschild()
schwarzschild = data_schwarzschild["metric"]
variables = data_schwarzschild["coordinates"]

print_geodesic_rhs(schwarzschild)

R = sympy.simplify(schwarzschild.compute_ricci_scalar())
print(R) # Get R = 0, which is because Schwarazschild is a vacuum solution

# The procedure is the same for Kerr 

### For FLRW 
data_flrw = getFLRW()
flrw = data_flrw["metric"]
variables = data_flrw["coordinates"]
a_t = data_flrw["parameters"]["a"]

# Computing the Ricci scalar and substituting a(t) = t^(2/3), corresponding to 
# an Einstein-de Sitter universe
R = sympy.simplify(flrw.compute_ricci_scalar())
t = variables[0]
R_substituted = R.subs(a_t, t**(2/3)).doit()
print(sympy.simplify(R_substituted)) 