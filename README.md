# Geodesic_EOM_deriver
Takes a metric and uses the Sympy Python package to derive the equations of motion along a geodesic.

These equations of motion are the components of the geodesic equation

<img src="https://latex.codecogs.com/svg.image?\large%20\frac{d^2x^\mu}{d\lambda^2}%20=%20-\Gamma^\mu_{\rho%20\sigma}%20\frac{dx^\rho}{d\lambda}%20\frac{dx^\sigma}{d\lambda}" />

The right hand side referred to in the code is thus the right hand side of the above equation, so the second derivative of <img src="https://latex.codecogs.com/svg.image?\large%20x^\mu" /> with respect to <img src="https://latex.codecogs.com/svg.image?\large%20\lambda" />. The metric given as input to the instances of the Metric class is <img src="https://latex.codecogs.com/svg.image?\large%20g_{\mu%20\nu}" />.

metric.py contains the class definition and all the code for computing the right hand side of the geodesic equation, and geodesic_eq_print.py has a short snippet for printing the components of the geodesic equation for both a Schwarzschild and Kerr metric, and both in Python and LaTeX syntax.
