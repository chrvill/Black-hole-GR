import numpy as np
import matplotlib.pyplot as plt 
from scipy import interpolate 

img_prefix = "images/"
txt_prefix = "txtfiles/"

c = 2.99792458e8     # Speed of light
k = 1.3806504e-23    # Boltzmann's constant
h = 6.62607015e-34   # Planck's constant

def compute_XYZ_to_RGB_matrix():
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

# Computing the XYZ_to_RGB matrix so that it can be used later
XYZ_to_RGB = compute_XYZ_to_RGB_matrix()

def color_matching_functions(lambdas):
    # https://www.researchgate.net/publication/337588486_Plotting_Colors_on_Color_Circle_Interconversion_between_XYZ_Values_and_RGB_Color_System
    color_matching_functions_table = np.loadtxt(f"{txt_prefix}cie_lookup_table.txt")[:, :4]

    table_lambdas, xbar, ybar, zbar = np.transpose(color_matching_functions_table)

    x_spline = interpolate.splrep(table_lambdas, xbar)
    xbar_interp = interpolate.splev(lambdas*1e9, x_spline)

    y_spline = interpolate.splrep(table_lambdas, ybar)
    ybar_interp = interpolate.splev(lambdas*1e9, y_spline)

    z_spline = interpolate.splrep(table_lambdas, zbar)
    zbar_interp = interpolate.splev(lambdas*1e9, z_spline)

    return xbar_interp, ybar_interp, zbar_interp

def compute_XYZ(lambdas, I):
    """
    Computes the XYZ values from the intensity distribution I(lambdas)
    """
    xbar, ybar, zbar = color_matching_functions(lambdas)

    # Integrating the intensity distribution * the color matching functions
    X = np.trapz(I*xbar, lambdas)
    Y = np.trapz(I*ybar, lambdas)
    Z = np.trapz(I*zbar, lambdas)

    return X, Y, Z

def compute_XYZ_from_T(T):
    """
    Computes the XYZ values from the blackbody distribution at temperature T
    """
    lambdas = np.linspace(380, 780, 1000)*1e-9

    I = blackbody_distribution(lambdas, T)

    return compute_XYZ(lambdas, I)

def RGB_from_XYZ(X, Y, Z):
    """
    Calculates the sRGB values from XYZ values
    """
    x = X/(X + Y + Z)
    y = Y/(X + Y + Z)
    z = 1 - x - y

    # Assuming Y = 1, and calculating the corresponding X and Z
    Y = 1
    X = x/y
    Z = z/y

    # The linear sRGB values
    rgb = np.matmul(XYZ_to_RGB, np.transpose([X, Y, Z]))

    if np.max(rgb) > 1:
        rgb /= np.max(rgb)

    # Making sure the RGB values are always non-negative
    return rgb.clip(min = 0)

def RGB_from_spectrum(lambdas, I):
    """
    Calculates the sRGB values from the intensity distribution I(lambdas)
    """
    X, Y, Z = compute_XYZ(lambdas, I)

    return RGB_from_XYZ(X, Y, Z)

def blackbody_distribution(lambdas, T):
    """
    Computes the blackbody distribution for the specified wavelengths and temperature
    """
    return lambdas**(-5)/(np.exp(h*c/(lambdas*k*T)) - 1)

def RGB_from_T(T):
    """
    Calculates the sRGB values directly from a blackbody distribution
    """
    lambdas = np.linspace(380, 780, 1000)*1e-9

    blackbody_dist = blackbody_distribution(lambdas, T)

    return RGB_from_spectrum(lambdas, blackbody_dist)

def plot_temperature_redshift_map(temperatures, redshifts):
    """
    Plots the sRGB values of the blackbody distributions for the specified temperatures
    and redshifts
    """
    fig, ax = plt.subplots()

    colors = np.zeros((len(temperatures), len(redshifts), 3))

    for i, T in enumerate(temperatures):
        for j, z in enumerate(redshifts):
            # Passing in the redshifted temperature
            rgb = RGB_from_T(T/(1 + z))

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

    fig.savefig(f"{img_prefix}temp_redshift_colormap.pdf", bbox_inches = "tight")
    
def plot_CIE_spectral_curve(ax):
    """
    Plots the spectral curve in the chromaticity diagram
    """
    lambdas = np.linspace(450, 700, 10000)*1e-9

    X, Y, Z = color_matching_functions(lambdas)

    x = X/(X + Y + Z)
    y = Y/(X + Y + Z)

    # The RGB color of each wavelength
    rgb = np.array([RGB_from_XYZ(X[i], Y[i], Z[i]) for i in range(len(lambdas))])

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

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    
def plot_planckian_locus(temperatures, ax):
    """
    Plots the Planckian locus for the specified temperatures
    """
    lambdas = np.linspace(380, 780, 1000)*1e-9

    X, Y, Z = np.transpose([compute_XYZ_from_T(T) for T in temperatures])

    RGB = np.array([RGB_from_XYZ(X[i], Y[i], Z[i]) for i in range(len(temperatures))])

    x = X/(X + Y + Z)
    y = Y/(X + Y + Z)

    for i in range(len(temperatures)):
        ax.plot(x[i], y[i], "o", color = RGB[i], alpha = 0.9, markersize = 5)
        
def plot_chromaticity_diagram(temperatures):
    """
    Plots the chromaticity diagram, including the spectral locus and Planckian locus
    """
    fig, ax = plt.subplots()

    plot_CIE_spectral_curve(ax)
    plot_planckian_locus(temperatures, ax)

    fig.savefig(f"{img_prefix}chromaticity_diagram_with_planckian_locus.pdf", bbox_inches = "tight")

def plot_color_matching_functions():
    """
    Plots the color matching functions as functions of wavelength
    """
    fig, ax = plt.subplots()

    lambdas = np.linspace(380, 780, 10000)*1e-9

    xbar, ybar, zbar = color_matching_functions(lambdas)

    ax.plot(lambdas*1e9, xbar, "r-", label = r"$\bar{x}$")
    ax.plot(lambdas*1e9, ybar, "g-", label = r"$\bar{y}$")
    ax.plot(lambdas*1e9, zbar, "b-", label = r"$\bar{z}$")

    ax.legend()
    ax.set_xlabel(r"$\lambda$ [nm]")

    fig.savefig(f"{img_prefix}color_matching_functions.pdf", bbox_inches = "tight")
    
if __name__ == "__main__":
    T = np.linspace(200, 10000, 100)
    z = np.linspace(-0.9, 1, 100)
    plot_temperature_redshift_map(T, z)
    plot_chromaticity_diagram(np.linspace(200, 10000, 1000))
    plot_color_matching_functions()

    fig, ax = plt.subplots()
    lambdas = np.linspace(100, 3000, 10000)*1e-9

    I1 = blackbody_distribution(lambdas, 3000)
    I2 = blackbody_distribution(lambdas, 7000)

    ax.plot(lambdas*1e9, I1/np.max(I1), "b-", label = r"$T = 3000$ K")
    ax.plot(lambdas*1e9, I2/np.max(I2), "r-", label = r"$T = 7000$ K")
    ax.set_xlabel(r"$\lambda$ [nm]")
    ax.set_ylabel(r"$I_\lambda/I_\lambda^\mathrm{max}$")

    ax.legend()

    fig.savefig(f"{img_prefix}blackbody_distribution.pdf", bbox_inches = "tight")