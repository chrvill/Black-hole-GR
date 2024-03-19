import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

c   = 2.99792458e8     # Speed of light
k_B = 1.3806504e-23    # Boltzmann's constant
h   = 6.62607015e-34   # Planck's constant

def blackbody_distribution(lambdas, T):
    """
    Computes the blackbody distribution for the specified wavelengths and temperature
    """
    return lambdas**(-5)/(np.exp(h*c/(lambdas*k_B*T)) - 1)

class ColorHandler:
    def __init__(self):
        color_matching_functions_table = np.loadtxt("txtfiles/cie_lookup_table.txt")[:, :4]
        self.table_lambdas, self.xbar, self.ybar, self.zbar = np.transpose(color_matching_functions_table)

        self.xbar_spline = interpolate.splrep(self.table_lambdas, self.xbar)
        self.ybar_spline = interpolate.splrep(self.table_lambdas, self.ybar)
        self.zbar_spline = interpolate.splrep(self.table_lambdas, self.zbar)

        self.XYZ_to_RGB = self.compute_XYZ_to_RGB_matrix()

    def compute_XYZ_to_RGB_matrix(self):
        """
        Defines the (x, y) coordinates of the red, green and blue primary colors of
        the RGB gamut. This follows the description in
        https://faculty.kfupm.edu.sa/ics/lahouari/Teaching/colorspacetransform-1.0.pdf
        """

        x_r = 0.64
        y_r = 0.33
        z_r = 1 - x_r - y_r

        #x_g = 0.29
        x_g = 0.3
        y_g = 0.6
        z_g = 1 - x_g - y_g

        x_b = 0.15
        y_b = 0.06
        z_b = 1 - x_b - y_b

        x_w = 0.3127
        y_w = 0.3290
        z_w = 1 - x_w - y_w
        Y_w = 1

        Y_r, Y_g, Y_b = 0.2126, 0.7152, 0.0722

        # Calculate the X- and Z-values of the primaries
        X_r = Y_r/y_r*x_r
        X_g = Y_g/y_g*x_g
        X_b = Y_b/y_b*x_b

        Z_r = Y_r/y_r*z_r
        Z_g = Y_g/y_g*z_g
        Z_b = Y_b/y_b*z_b

        # The matrix transforming XYZ to linear RGB
        XYZ_to_RGB = np.array([[X_r, X_g, X_b], [Y_r, Y_g, Y_b], [Z_r, Z_g, Z_b]])

        return np.linalg.inv(XYZ_to_RGB)

    def color_matching_functions(self, lambdas):
        xbar_interp = interpolate.splev(lambdas*1e9, self.xbar_spline)
        ybar_interp = interpolate.splev(lambdas*1e9, self.ybar_spline)
        zbar_interp = interpolate.splev(lambdas*1e9, self.zbar_spline)

        return xbar_interp, ybar_interp, zbar_interp

    def compute_XYZ_spectrum(self, I, lambdas):
        xbar, ybar, zbar = self.color_matching_functions(lambdas)

        X = np.trapz(I*xbar, lambdas)
        Y = np.trapz(I*ybar, lambdas)
        Z = np.trapz(I*zbar, lambdas)

        return X, Y, Z

    def compute_XYZ_blackbody(self, T):
        lambdas = np.linspace(380, 780, 1000)*1e-9
        blackbody_dist = blackbody_distribution(lambdas, T)

        return self.compute_XYZ_spectrum(blackbody_dist, lambdas)

    def RGB_from_XYZ(self, X, Y, Z):
        x = X/(X + Y + Z)
        y = Y/(X + Y + Z)
        z = 1 - x - y

        # Assuming Y = 1, and calculating the corresponding X and Z
        Y = 1
        X = x/y
        Z = z/y

        # The linear sRGB values
        rgb = np.matmul(self.XYZ_to_RGB, np.transpose([X, Y, Z]))

        if np.max(rgb) > 1:
            rgb /= np.max(rgb)

        # Making sure the RGB values are always non-negative
        return rgb.clip(min = 0)

    def RGB_monochromatic(self, lambda0):
        X, Y, Z = self.color_matching_functions(lambda0)

        rgb = self.RGB_from_XYZ(X, Y, Z)
        return rgb

    def RGB_from_T(self, T):
        X, Y, Z = self.compute_XYZ_blackbody(T)

        rgb = self.RGB_from_XYZ(X, Y, Z)
        return rgb

    def RGB_from_spectrum(self, I, lambdas):
        X, Y, Z = self.compute_XYZ_spectrum(I, lambdas)
        return self.RGB_from_XYZ(X, Y, Z)

def plot_temperature_redshift_map(temperatures, redshifts, colorhandler):
    fig, ax = plt.subplots()

    colors = np.zeros((len(temperatures), len(redshifts), 3))

    for i, T in enumerate(temperatures):
        for j, z in enumerate(redshifts):
            # Passing in the redshifted temperature
            rgb = colorhandler.RGB_from_T(T/(1 + z))

            colors[i, j] = rgb

    ax.imshow(colors, interpolation = "gaussian")

    xticks = np.arange(-0.8, 1.2, 0.2)
    yticks = np.arange(1000, 11000, 1000)

    xtick_indices = [np.argmin(np.abs(redshifts - xticks[i])) for i in range(len(xticks))]
    ytick_indices = [np.argmin(np.abs(temperatures - yticks[i])) for i in range(len(yticks))]

    ax.set_xticks(xtick_indices)
    ax.set_xticklabels(np.round(xticks + 1, 1))

    ax.set_yticks(ytick_indices)
    ax.set_yticklabels(yticks)

    ax.invert_yaxis()
    ax.set_xlabel(r"$1 + z$")
    ax.set_ylabel(r"$T$ [K]")

    fig.savefig("images/temp_redshift_colormap.pdf", bbox_inches = "tight")

def plot_chromaticity_diagram(colorhandler, with_planckian_locus = True):
    """
    Plots the spectral curve in the chromaticity diagram
    """
    fig, ax = plt.subplots()

    lambdas = np.linspace(450, 700, 10000)*1e-9

    X, Y, Z = colorhandler.color_matching_functions(lambdas)

    x = X/(X + Y + Z)
    y = Y/(X + Y + Z)

    # The RGB color of each wavelength
    rgb = np.array([colorhandler.RGB_from_XYZ(X[i], Y[i], Z[i]) for i in range(len(lambdas))])

    s = 10

    for i in range(0, len(lambdas) - s, s):
        ax.plot(x[i: i + s + 1], y[i: i + s + 1], color = rgb[i])

    # Points on the locus that we want to mark
    points_on_locus = np.arange(460, 640, 20)
    indices = [np.argmin(np.abs(lambdas*1e9 - wavelength)) for wavelength in points_on_locus]

    for i in indices:
        ax.text(x[i] + 0.01, y[i] + 0.01, "{:.0f} nm".format(lambdas[i]*1e9))
        ax.plot(x[i], y[i], "ko")

    ax.set_xlim(-0.05, 0.85)

    #ax.plot(x, y, "k-")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    if with_planckian_locus:
        temperatures = np.linspace(200, 10000, 100)
        lambdas = np.linspace(380, 780, 1000)*1e-9

        X, Y, Z = np.transpose([colorhandler.compute_XYZ_blackbody(T) for T in temperatures])

        RGB = np.array([colorhandler.RGB_from_XYZ(X[i], Y[i], Z[i]) for i in range(len(temperatures))])

        x = X/(X + Y + Z)
        y = Y/(X + Y + Z)

        for i in range(len(temperatures)):
            ax.plot(x[i], y[i], "o", color = RGB[i], alpha = 0.9, markersize = 5)

    fig.savefig("images/chromaticity_diagram.pdf", bbox_inches = "tight")

if __name__ == "__main__":
    colorhandler = ColorHandler()
    plot_chromaticity_diagram(colorhandler)

    T = np.linspace(200, 10000, 100)
    z = np.linspace(-0.9, 1, 100)
    plot_temperature_redshift_map(T, z, colorhandler)

    R = np.zeros_like(T)
    G = np.zeros_like(T)
    B = np.zeros_like(T)

    for i in range(len(T)):
        rgb = colorhandler.RGB_from_T(T[i])
        R[i] = rgb[0]
        G[i] = rgb[1]
        B[i] = rgb[2]

    fig, ax = plt.subplots()
    ax.plot(T, R, "r-")
    ax.plot(T, G, "g-")
    ax.plot(T, B, "b-")

    fig.savefig("images/blackbody_RGB.pdf")
