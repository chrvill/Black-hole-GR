import numpy as np 
import matplotlib.pyplot as plt 

"""
Assuming the only non-zero off-diagonal components of the metric are g_tph and g_pht. 
Will always work under the assumption of spherical symmetry.   
"""

class GeodesicSolver:
    def __init__(self, parameters):
        self.parameters = parameters
        
    def g_tt(self, r, theta):
        return 0  
    
    def g_rr(self, r, theta):
        return 0
    
    def g_thth(self, r, theta):
        return 0
    
    def g_phph(self, r, theta):
        return 0  
    
    def g_tph(self, r, theta):
        return 0 
    
    def compute_norm(self, r, theta, p0, p1, p2, p3):
        """
        Calculating the norm of a four-vector with components p0, p1, p2 and p3 at position given by r and theta (phi does not matter due to axial symmetry)
        """
        return self.g_tt(r, theta)*p0**2 + self.g_rr(r, theta)*p1**2 + self.g_thth(r, theta)*p2**2 \
             + self.g_phph(r, theta)*p3**2 + 2*self.g_tph(r, theta)*p0*p3

    def compute_p0(self, r, theta, p1, p2, p3, mu = 0):
        """
        Calculates the 0th component of the four-momentum of a particle
        given the 1st, 2nd and 3rd components. mu is the value of
        the contraction of the four-momentum with itself - for photons it is
        equal to 0, and for massive particles -1.
        """
        g_tt   = self.g_tt(r, theta)
        g_tph  = self.g_tph(r, theta)
        g_rr   = self.g_rr(r, theta)
        g_thth = self.g_thth(r, theta)
        g_phph = self.g_phph(r, theta)

        # Inside the ergosphere g_tt turns positive, so there we have to use the - sign in the +/- from
        # the quadratic equation rather than the + sign we use outside the ergosphere.
        if g_tt < 0:
            return -g_tph/g_tt*p3 + np.sqrt((g_tph/g_tt)**2*p3**2 - 1/g_tt*(g_rr*p1**2 + g_thth*p2**2 + g_phph*p3**2 - mu))
        else:
            return -g_tph/g_tt*p3 - np.sqrt((g_tph/g_tt)**2*p3**2 - 1/g_tt*(g_rr*p1**2 + g_thth*p2**2 + g_phph*p3**2 - mu))

    def geodesic_eq_t(self, y):
        pass 
    
    def geodesic_eq_r(self, y):
        pass
    
    def geodesic_eq_theta(self, y):
        pass
    
    def geodesic_eq_phi(self, y):
        pass
    
    def geodesic_eq_rhs(self, y):
        """
        Calculates the right-hand side of the geodesic equation, so the
        derivatives of all the coordinates and components of the four-momentum
        of our particle with respect to the affine parameter along the geodesic.
        """
        t, r, theta, phi, p0, p1, p2, p3 = y

        # Initializing the derivatives of the coordinates and four-momentum of our particle
        derivatives = np.zeros_like(y)

        derivatives[0] = p0
        derivatives[1] = p1
        derivatives[2] = p2
        derivatives[3] = p3
        
        # The extremely nasty right hand sides of the geodesic equation, derived using metric.py
        derivatives[4] = self.geodesic_eq_t(y)
        derivatives[5] = self.geodesic_eq_r(y)
        derivatives[6] = self.geodesic_eq_theta(y)
        derivatives[7] = self.geodesic_eq_phi(y)
        
        return derivatives
    
    def RKF45(self, y, h, tol = 1e-7):
        abs_error = np.inf

        while abs_error > tol:
            """
            The integration scheme itself
            """
            k1 = self.geodesic_eq_rhs(y)
            k2 = self.geodesic_eq_rhs(y +       1/4*k1*h)
            k3 = self.geodesic_eq_rhs(y +      3/32*k1*h +      9/32*k2*h)
            k4 = self.geodesic_eq_rhs(y + 1932/2197*k1*h - 7200/2197*k2*h + 7296/2197*k3*h)
            k5 = self.geodesic_eq_rhs(y +   439/216*k1*h -         8*k2*h +  3680/513*k3*h -  845/4104*k4*h)
            k6 = self.geodesic_eq_rhs(y -      8/27*k1*h +         2*k2*h - 3544/2565*k3*h + 1859/4104*k4*h - 11/40*k5*h)

            # The value of y on the next step
            new_y = y + 16/135*k1*h + 6656/12825*k3*h + 28561/56430*k4*h - 9/50*k5*h + 2/55*k6*h

            # The numerical error associated with the RKF45 scheme
            error = -1/360*k1*h + 128/4275*k3*h + 2197/75240*k4*h - 1/50*k5*h - 2/55*k6*h

            # The "Pythagorean" length of the error
            abs_error = np.sqrt(np.sum(error**2))

            # Calculating a new stepsize based on the current error
            new_h = 0.9*h*(tol/abs_error)**(1/5)
            h = new_h
            #h = np.min([new_h, 1e-1])

        return new_y, h
    
    def solve(self, y0, n, h0):
        """
        Performs the actual solving of the equations of motion by calling the RungeKuttaFehlberg function.
        y0 contains the initial values for the variables for which we solve the geodesic equation. It is assumed
        to take the form np.array([t, r, theta, phi, p0, p1, p2, p3]), where p0-p3 are the components of
        the four-momentum of the particle for which we solve the geodesic equation.
        """
        # The initial value of the affine parameter. Can be taken to be zero without loss of generality.
        # The steps we perform along the geodesic are in terms of this affine parameter
        affine_parameter = np.zeros(n)

        # Initializing the list that will be storing the position and four-momentum at each step
        y = np.zeros((n, len(y0)))
        y[0] = y0
        h = h0 # Initial step size

        # Performing the numerical integration
        for i in range(n - 1):
            y_next, h = self.RKF45(y[i], h)

            y[i + 1] = y_next
            affine_parameter[i + 1] = affine_parameter[i] + h

        return affine_parameter, y
    
    def plotGeodesic(self, geodesic):
        """
        Plots a geodesic in two dimensions. The geodesic array is assumed to be of the following form
            t = geodesic[:, 0]
            r = geodesic[:, 1]
            theta = geodesic[:, 2]
            phi = geodesic[:, 3]
        and
            p_i = geodesic[:, i + 4]
        """
        pass 
    
    def compute_redshift(self, geodesic, u_observer, u_emitter):
        """
        Calculates the redshift of a photon emitted by an observer with four-velocity u_emitter
        and received by an observer with four-velocity u_observer. The photon travels along a geodesic defined by
        the geodesic-array. geodesic[0] represents the emission point and geodesic[-1] represents the observation point.
        """
        r_init, theta_init, phi_init, p0_init, p1_init, p2_init, p3_init = geodesic[0, 1:]
        r_final, theta_final, phi_final, p0_final, p1_final, p2_final, p3_final = geodesic[-1, 1:]

        # Emission wavelength (to be precise the wavelength is only proportional to this, we're missing physical constants)
        lambda_init = -(1/(self.g_tt(r_init, theta_init)*p0_init*u_emitter[0] \
                         + self.g_rr(r_init, theta_init)*p1_init*u_emitter[1] \
                         + self.g_thth(r_init, theta_init)*p2_init*u_emitter[2] \
                         + self.g_phph(r_init, theta_init)*p3_init*u_emitter[3] \
                         + self.g_tph(r_init, theta_init)*(p0_init*u_emitter[3] + p3_init*u_emitter[0])))

        # Observed wavelength
        lambda_final = -(1/(self.g_tt(r_final, theta_final)*p0_final*u_observer[0] \
                          + self.g_rr(r_final, theta_final)*p1_final*u_observer[1] \
                          + self.g_thth(r_final, theta_final)*p2_final*u_observer[2] \
                          + self.g_phph(r_final, theta_final)*p3_final*u_observer[3] \
                          + self.g_tph(r_final, theta_final)*(p0_final*u_observer[3] + p3_final*u_observer[0])))

        return lambda_final/lambda_init