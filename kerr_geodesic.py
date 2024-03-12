import numpy as np
import matplotlib.pyplot as plt

class KerrMetric:
    def __init__(self, a, M = 1):
        self.a = a
        self.M = M

        self.Rs = 1 + np.sqrt(1 - self.a**2)

    """
    The covariant components of the metric tensor
    """
    def g_tt(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2

        return -(1 - 2*r/sigma)

    def g_tph(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2

        return -2*self.a*r/sigma*np.sin(theta)**2

    def g_rr(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2
        delta = r**2 - 2*self.M*r + self.a**2

        return sigma/delta

    def g_thth(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2

        return sigma

    def g_phph(self, r, theta):
        sigma = r**2 + self.a**2*np.cos(theta)**2
        delta = r**2 - 2*self.M*r + self.a**2
        Lambda = (r**2 + self.a**2)**2 - self.a**2*delta*np.sin(theta)**2

        return Lambda/sigma*np.sin(theta)**2

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

        cos = np.cos(theta)
        sin = np.sin(theta)

        a = self.a

        # The extremely nasty right hand sides of the geodesic equation, derived using metric.py
        derivatives[4] = -2*p0*p1*(-1.0*a*r*(4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 - 2*a*sin**2/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(-4*r**2/(a**2*cos**2 + r**2)**2 + 2/(a**2*cos**2 + r**2))*(-a**4*cos**2 - a**2*r**2*cos**2 - a**2*r**2 - 2*a**2*r*sin**2 - r**4)/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) - 2*p0*p2*(2.0*a**2*r*(-a**4*cos**2 - a**2*r**2*cos**2 - a**2*r**2 - 2*a**2*r*sin**2 - r**4)*sin*cos/((a**2*cos**2 + r**2)**2*(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) - 1.0*a*r*(-4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 - 4*a*r*sin*cos/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) - 2*p1*p3*(-1.0*a*r*(-4*a**2*r**2*sin**2/(a**2*cos**2 + r**2)**2 + 2*a**2*sin**2/(a**2*cos**2 + r**2) + 2*r)*sin**2/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 - 2*a*sin**2/(a**2*cos**2 + r**2))*(-a**4*cos**2 - a**2*r**2*cos**2 - a**2*r**2 - 2*a**2*r*sin**2 - r**4)/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) - 2*p2*p3*(-1.0*a*r*((4*a**4*r*sin**3*cos/(a**2*cos**2 + r**2)**2 + 4*a**2*r*sin*cos/(a**2*cos**2 + r**2))*sin**2 + 2*(2*a**2*r*sin**2/(a**2*cos**2 + r**2) + a**2 + r**2)*sin*cos)/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(-4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 - 4*a*r*sin*cos/(a**2*cos**2 + r**2))*(-a**4*cos**2 - a**2*r**2*cos**2 - a**2*r**2 - 2*a**2*r*sin**2 - r**4)/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3))

        derivatives[5] = 2.0*a**2*p1*p2*sin*cos/(a**2*cos**2 + r**2) + 1.0*r*p2**2*(a**2 + r**2 - 2*r)/(a**2*cos**2 + r**2) - 0.5*p0**2*(4*r**2/(a**2*cos**2 + r**2)**2 - 2/(a**2*cos**2 + r**2))*(a**2 + r**2 - 2*r)/(a**2*cos**2 + r**2) - 1.0*p0*p3*(-4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 + 2*a*sin**2/(a**2*cos**2 + r**2))*(a**2 + r**2 - 2*r)/(a**2*cos**2 + r**2) - 0.5*p1**2*(2*r/(a**2 + r**2 - 2*r) + (2 - 2*r)*(a**2*cos**2 + r**2)/(a**2 + r**2 - 2*r)**2)*(a**2 + r**2 - 2*r)/(a**2*cos**2 + r**2) + 0.5*p3**2*(a**2 + r**2 - 2*r)*(-4*a**2*r**2*sin**2/(a**2*cos**2 + r**2)**2 + 2*a**2*sin**2/(a**2*cos**2 + r**2) + 2*r)*sin**2/(a**2*cos**2 + r**2)

        derivatives[6] = 2.0*a**2*r*p0**2*sin*cos/(a**2*cos**2 + r**2)**3 - 1.0*a**2*p1**2*sin*cos/((a**2*cos**2 + r**2)*(a**2 + r**2 - 2*r)) + 1.0*a**2*p2**2*sin*cos/(a**2*cos**2 + r**2) - 2.0*r*p1*p2/(a**2*cos**2 + r**2) - 1.0*p0*p3*(4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 + 4*a*r*sin*cos/(a**2*cos**2 + r**2))/(a**2*cos**2 + r**2) - 0.5*p3**2*(-(4*a**4*r*sin**3*cos/(a**2*cos**2 + r**2)**2 + 4*a**2*r*sin*cos/(a**2*cos**2 + r**2))*sin**2 - 2*(2*a**2*r*sin**2/(a**2*cos**2 + r**2) + a**2 + r**2)*sin*cos)/(a**2*cos**2 + r**2)

        derivatives[7] = -2*p0*p1*(-1.0*a*r*(-4*r**2/(a**2*cos**2 + r**2)**2 + 2/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 - 2*a*sin**2/(a**2*cos**2 + r**2))*(a**2*cos**2 + r**2 - 2*r)/(a**4*sin**2*cos**2 + a**2*r**2*sin**2*cos**2 + a**2*r**2*sin**2 + 2*a**2*r*sin**4 - 2*a**2*r*sin**2 + r**4*sin**2 - 2*r**3*sin**2)) - 2*p0*p2*(-4.0*a**3*r**2*sin*cos/((a**2*cos**2 + r**2)**2*(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3)) + 0.5*(-4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 - 4*a*r*sin*cos/(a**2*cos**2 + r**2))*(a**2*cos**2 + r**2 - 2*r)/(a**4*sin**2*cos**2 + a**2*r**2*sin**2*cos**2 + a**2*r**2*sin**2 + 2*a**2*r*sin**4 - 2*a**2*r*sin**2 + r**4*sin**2 - 2*r**3*sin**2)) - 2*p1*p3*(-1.0*a*r*(4*a*r**2*sin**2/(a**2*cos**2 + r**2)**2 - 2*a*sin**2/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*(a**2*cos**2 + r**2 - 2*r)*(-4*a**2*r**2*sin**2/(a**2*cos**2 + r**2)**2 + 2*a**2*sin**2/(a**2*cos**2 + r**2) + 2*r)*sin**2/(a**4*sin**2*cos**2 + a**2*r**2*sin**2*cos**2 + a**2*r**2*sin**2 + 2*a**2*r*sin**4 - 2*a**2*r*sin**2 + r**4*sin**2 - 2*r**3*sin**2)) - 2*p2*p3*(-1.0*a*r*(-4*a**3*r*sin**3*cos/(a**2*cos**2 + r**2)**2 - 4*a*r*sin*cos/(a**2*cos**2 + r**2))/(a**4*cos**2 + a**2*r**2*cos**2 + a**2*r**2 + 2*a**2*r*sin**2 - 2*a**2*r + r**4 - 2*r**3) + 0.5*((4*a**4*r*sin**3*cos/(a**2*cos**2 + r**2)**2 + 4*a**2*r*sin*cos/(a**2*cos**2 + r**2))*sin**2 + 2*(2*a**2*r*sin**2/(a**2*cos**2 + r**2) + a**2 + r**2)*sin*cos)*(a**2*cos**2 + r**2 - 2*r)/(a**4*sin**2*cos**2 + a**2*r**2*sin**2*cos**2 + a**2*r**2*sin**2 + 2*a**2*r*sin**4 - 2*a**2*r*sin**2 + r**4*sin**2 - 2*r**3*sin**2))

        return derivatives

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

def RungeKuttaFehlberg(y, h, rhs, tol = 1e-7):
    """
    Uses the Runge-Kutta-Fehlberg (RKF45) integration scheme to step our particle forward
    in time along a geodesic. y contains our the functions that we wish to step forward in time.
    h is the step size, rhs is the right hand side, kerr is an instance of the KerrMetric class,
    tol is an error tolerance level that we choose.
    """

    # Initializing an error term
    abs_error = np.inf

    while abs_error > tol:
        """
        The integration scheme itself
        """
        k1 = rhs(y)
        k2 = rhs(y +       1/4*k1*h)
        k3 = rhs(y +      3/32*k1*h +      9/32*k2*h)
        k4 = rhs(y + 1932/2197*k1*h - 7200/2197*k2*h + 7296/2197*k3*h)
        k5 = rhs(y +   439/216*k1*h -         8*k2*h +  3680/513*k3*h -  845/4104*k4*h)
        k6 = rhs(y -      8/27*k1*h +         2*k2*h - 3544/2565*k3*h + 1859/4104*k4*h - 11/40*k5*h)

        # The value of y on the next step
        new_y = y + 16/135*k1*h + 6656/12825*k3*h + 28561/56430*k4*h - 9/50*k5*h + 2/55*k6*h

        # The numerical error associated with the RKF45 scheme
        error = -1/360*k1*h + 128/4275*k3*h + 2197/75240*k4*h - 1/50*k5*h - 2/55*k6*h

        # The "Pythagorean" lenght of the error
        abs_error = np.sqrt(np.sum(error**2))

        # Calculating a new stepsize based on the current error
        new_h = 0.9*h*(tol/abs_error)**(1/5)
        h = new_h

    return new_y, h

def solve(y0, n, h0, kerr):
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
        y_next, h = RungeKuttaFehlberg(y[i], h, kerr.geodesic_eq_rhs, tol = 1e-7)

        y[i + 1] = y_next
        affine_parameter[i + 1] = affine_parameter[i] + h

    return affine_parameter, y

def compute_EH_and_ergosphere(kerr):
    """
    Calculates the positions of the outer event horizon and ergosphere.
    kerr is an instance of the KerrMetric class.
    """
    theta = np.linspace(0, 2*np.pi, 100)
    x_EH = np.sqrt(kerr.Rs**2 + kerr.a**2)*np.cos(theta)
    y_EH = np.sqrt(kerr.Rs**2 + kerr.a**2)*np.sin(theta)

    x_ergo = np.sqrt(2**2 + kerr.a**2)*np.cos(theta)
    y_ergo = np.sqrt(2**2 + kerr.a**2)*np.sin(theta)

    return x_EH, y_EH, x_ergo, y_ergo

def plotGeodesic2d(solution, kerr):
    """
    Plots a geodesic in two dimensions. The solution array is assumed to be of the form where
        t = solution[:, 0]
        r = solution[:, 1]
        theta = solution[:, 2]
        phi = solution[:, 3]
    and
        p_i = solution[:, i + 4]

    kerr is an instance of the KerrMetric class.
    """
    r = solution[:, 1]
    theta = solution[:, 2]
    phi = solution[:, 3]

    x = np.sqrt(r**2 + kerr.a**2)*np.cos(phi)*np.sin(theta)
    y = np.sqrt(r**2 + kerr.a**2)*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)

    fig, ax = plt.subplots()
    ax.plot(x, y, "b-", label = "Geodesic")
    ax.plot(x[0], y[0], "ro", label = "Start")

    EH = plt.Circle((0, 0), np.sqrt(kerr.Rs**2 + kerr.a**2), color = "k")
    ax.add_patch(EH)

    circle_angles = np.linspace(0, 2*np.pi, 100)
    x_ergo = np.sqrt(2**2 + kerr.a**2)*np.cos(circle_angles)
    y_ergo = np.sqrt(2**2 + kerr.a**2)*np.sin(circle_angles)
    ax.plot(x_ergo, y_ergo, "r--", label = "Ergosphere")

    ax.set_xlabel(r"$x/M$")
    ax.set_ylabel(r"$y/M$")

    ax.legend()
    ax.set_aspect("equal")
    fig.savefig("images/geodesic.pdf", bbox_inches = "tight")

if __name__ == "__main__":
    # Initializing the Kerr metric
    a = 1
    kerr = KerrMetric(a = a)

    # Initial position of the particle in Boyer-Lindquist coordinates
    t0     = 0
    r0     = 10
    theta0 = np.pi/2
    phi0   = 0

    # Components of the initial four-momentum of the particle
    p1 = 0
    p2 = 0
    p3 = 0
    p0 = kerr.compute_p0(r0, theta0, p1, p2, p3, mu = -1)

    # Collecting the variables we will be interested in evolving in time into a single list
    y0 = [t0, r0, theta0, phi0, p0, p1, p2, p3]

    # Solve the equations of motion for a specified number of steps with a specified initial step size
    tau, solution = solve(y0, 150, 1e-7, kerr)

    plotGeodesic2d(solution, kerr)
    r_final = solution[-1, 1]
    theta_final = solution[-1, 2]

    # Defining four-velocities for the observer and emitter of the photon
    # Assuming the emitter is at rest
    u_em = np.array([kerr.compute_p0(r0, theta0, 0, 0, 0, mu = -1), 0, 0, 0])

    # Assuming the observer is a ZAMO
    u_obs = kerr.ZAMO_four_vel(r_final, theta_final)

    redshift = kerr.compute_redshift(solution, u_obs, u_em)
    print("1 + z = {}".format(redshift))
