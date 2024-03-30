# Black-hole-GR
Functionality for computing various quantities in general relativity (GR), and in particular quantities relevant to black holes. The main functionality is the derivation of explicit expressions for the components of the geodesic equation: 

<img src="https://latex.codecogs.com/svg.image?\color{white}\frac{d^2&space;x^\mu}{d\lambda^2}=-\Gamma^\mu_{\rho\sigma}\frac{dx^\rho}{d\lambda}\frac{dx^\sigma}{d\lambda}\quad\quad(1)" />

where <img src=https://latex.codecogs.com/svg.image?\inline\color{white}&space;\Gamma^\mu_{\rho\sigma}> are the Christoffel symbols of the metric <img src=https://latex.codecogs.com/svg.image?\inline\color{white}&space;g_{\mu\nu}>. These are given by 

<img src="https://latex.codecogs.com/svg.image?\color{white}\Gamma^\mu_{\rho\sigma}=\frac{1}{2}g^{\mu\nu}\left(\partial_\rho&space;g_{\sigma\nu}&plus;\partial_\sigma&space;g_{\rho\nu}-\partial_\nu&space;g_{\rho\sigma}\right)\quad\quad(2)" />

The code uses the Sympy library in Python to do symbolic differentiation and algebra. Once we have the Christoffel symbols it's very easy to compute the Riemann tensor and the Ricci tensor and scalar (at least easy in principle, the computations are tedious).

---
`metric.py` contains the code for deriving the explicit expressions for the components of the geodesic equation. The right hand side (rhs) referred to in the code is the right hand side of (1). It should be noted that the computation of the Christoffel symbols could be optimized by using the symmetry in the two lower indices, but that is not done here. `geodesic_eq_print` has short example snippets for how the code in `metric.py`can be used. This includes printing the geodesic equation both in Python and LaTeX syntax. 

`kerr_newman.py` is an extension of `metric.py`. The latter only describes spacetime without electromagnetic fields. But `kerr_newman.py` takes electromagnetic fields into account in the specific case of a charged, rotating black hole, which is described by the Kerr-Newman metric. 

`geodesic_solver.py` defines a general class `GeodesicSolver` for solving the geodesic equation for a given metric. The actual expressions for the metric components and the components of the geodesic equation are not defined in this code, but should instead be defined in classes which inherit from `GeodesicSolver`. This class uses the Runge-Kutta-Fehlberg algorithm to solve the geodesic equation numerically.  

`kerr_geodesic.py` and `schwarzschild_deSitter.py` ìnherit from `GeodesicSolver` to define classes for the Kerr metric and Schwarzschild-de Sitter metric respectively. The Kerr metric describes a chargeless, rotating black hole, while the Schwarzschild describes a universe with both a chargeless, non-rotating black hole and a cosmological constant.    

`blackbody_colors.py` is not really related to GR. It contains code for computing the sRGB color of blackbodies at arbitrary temperatures and redshifts. The reason for including this code in the repo is that this color calculation is essential for calculating colors in relativistic renderers. This is because relativistic redshift will cause the emitted spectrum of an object to be redshifted/blueshifted, which greatly impacts the color we perceive. 

Using a geodesic solver like `kerr_geodesic.py` with the geodesic equation derived using `metric.py` and calculating colors using something like `blackbody_colors.py` we can produce black hole renders like this

![black-hole-animation](images/black_hole_animation.gif)

Note, though, that this animation was not rendered using Python. It's rendered in VEX, but the functionality one has to implement to render a black hole is effectively what we have here. A whole lot of rendering-related aspects have to be in place for this to look good, but the foundation of any black hole renderer is a geodesic tracer like in `geodesic_solver.py`. 