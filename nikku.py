import numpy as np
from scipy.optimize import fsolve
import scipy.optimize as optimize
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
R = 270000  # Radius in meters
rho = 3500  # Density (kg/m^3)
yearins = 365.25 * 24 * 3600
mass = 4 / 3 * np.pi * R**3 * rho  # Mass in kg, for Vesta's radius and density
v = 4 / 3 * np.pi * R**3

# Aluminum and iron data
molar_mass_Al = 0.027  # kg/mol for aluminum
avogadro_number = 6.022e23  # atoms/mol
ratio_26Al_27Al = 5e-5  # Initial 26Al/27Al ratio
aluminum_content_mass_fraction = 0.0113  # Total aluminum content by mass
rho_Al = aluminum_content_mass_fraction * rho  # kg/m^3
N_27Al = (rho_Al / molar_mass_Al) * avogadro_number  # atoms/m^3
A0_26Al = ratio_26Al_27Al * N_27Al

molar_mass_Fe = 0.056  # kg/mol
ratio_60Fe_56Fe = 1e-8  # Initial 26Al/27Al ratio
Fe_content_mass_fraction = 0.24  # Total Fe content by mass
rho_Fe = Fe_content_mass_fraction * rho  # kg/m^3
N_Fe = (rho_Fe / molar_mass_Fe) * avogadro_number  # atoms/m^3
A0_60Fe = ratio_60Fe_56Fe * N_Fe

# Half-lives and decay constants
half_life_Al = 0.72e6
half_life_Fe = 1.5e6
lambda_Al = np.log(2) / half_life_Al
lambda_Fe = np.log(2) / half_life_Fe

# Decay energies in joules per atom
E_decay_Al = 3 * 1.60218e-13
E_decay_Fe = 3 * 1.60218e-13
t0 = 2.85e6  # Initial time offset in years

# Function for radiogenic heat production
def Qradnl(t):
    Q_Al = A0_26Al * E_decay_Al * np.exp(-lambda_Al * (t + t0))
    Q_Fe = A0_60Fe * E_decay_Fe * np.exp(-lambda_Fe * (t + t0))
    return Q_Al + Q_Fe

# Thermal parameters
k = 3.8e-5 * yearins  # Thermal conductivity (W/m/K)
cp = 2100e1  # Specific heat capacity (J/kg/K)
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
epsilon = 0.8  # Emissivity
T_neb = 292  # Nebular temperature (K)

# Radial and time grids
Nr = 300  # Number of radial nodes
Nt = 25000  # Number of time steps
t_f = 1.75e6  # Final time (seconds)
t = np.linspace(0, t_f, Nt)
dt = t_f / (Nt - 1)
r = np.linspace(0, R, Nr)
dr = R / (Nr - 1)
T = np.full(Nr, 290)  # Initial temperature (K)

# Initialize matrices
A = np.zeros((Nr, Nr))
B = np.zeros((Nr, Nr))

# Symmetry at the center (r=0)
A[0, 0] = 1
B[0, 0] = 1

# Precompute coefficients
alpha = k * dt / (2 * dr**2)
beta = lambda ri: k * dt / (2 * ri * dr)

for i in range(1, Nr - 1):
    A[i, i - 1] = -alpha + beta(r[i])
    A[i, i] = 1 + 2 * alpha
    A[i, i + 1] = -alpha - beta(r[i])
    B[i, i - 1] = alpha - beta(r[i])
    B[i, i] = 1 - 2 * alpha
    B[i, i + 1] = alpha + beta(r[i])

# Surface (r=R): Radiative boundary
A[-1, -2] = -k / dr
A[-1, -1] = k / dr + epsilon * sigma * 4 * T[-1]**3
B[-1, -2] = k / dr
B[-1, -1] = -k / dr 

# Time evolution

times_to_plot = [.25e6,.5e6, .75e6, 1e6, 1.25e6, 1.5e6]  # Times for plotting (in seconds)
temperature_profiles = []  # Store temperature profiles at specific times

tolerance = dt / 2  # Tolerance for time comparison

for n in tqdm(range(1, Nt)):
    b = B @ T + (Qradnl(t[n - 1]) - Qradnl(t[n])) * dt / (rho * cp)
    # Solve the boundary condition equation at the surface
    def f(u, u_prev, emissivity, sigma, T_nebula, dr, rho, cp_basalt):
        return u - u_prev + (emissivity * sigma * (u**4 - T_nebula**4) * dr) / (rho * cp_basalt*dr)

    T[-1] = fsolve(f, T[-2], args=(T[-2], epsilon, sigma, T_neb, dr, rho, cp))
     
    # Solve the linear system A * T_new = b
    T_new = np.linalg.solve(A, b)
    
    # Clamp the temperature to ensure T >= 290 K
    T_new = np.maximum(T_new, T_neb)
    
    T = T_new

    # Store temperatures at specific times
    if t[n] in times_to_plot:
        temperature_profiles.append(T.copy())

# Store temperatures at specific times
    for time in times_to_plot:
        if abs(t[n] - time) < tolerance:
            temperature_profiles.append(T.copy())
            break
    print(T)
# Plot temperature vs radius
plt.figure(figsize=(10, 6))
for i, time in enumerate(times_to_plot):
    plt.plot(r / 1e3, temperature_profiles[i], label=f"{(time / 1e6) +2.85:.2f} Myr")

plt.xlabel("Radius (km)")
plt.ylabel("Temperature (K)")
plt.title("Temperature vs. Radius")
plt.legend()
plt.show()