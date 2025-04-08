import numpy as np
import matplotlib.pyplot as plt

# Time vector: 24 hours with 500 sample points for smooth curves.
t = np.linspace(0, 24, 500)

# --- Define consumption functions for each building type ---

# Residential types:
def single_family_consumption(t):
    baseline = 0.2
    morning_peak = 1.0 * np.exp(-((t - 8) / 1.5) ** 2)
    evening_peak = 1.2 * np.exp(-((t - 19) / 1.5) ** 2)
    return baseline + morning_peak + evening_peak

def apartment_consumption(t):
    baseline = 0.3
    morning_peak = 0.8 * np.exp(-((t - 8) / 1.2) ** 2)
    evening_peak = 0.9 * np.exp(-((t - 19) / 1.2) ** 2)
    return baseline + morning_peak + evening_peak

def restaurant_consumption(t):
    baseline = 0.1
    lunch_peak = 1.5 * np.exp(-((t - 12) / 0.8) ** 2)
    dinner_peak = 2.0 * np.exp(-((t - 20) / 1.0) ** 2)
    return baseline + lunch_peak + dinner_peak

def office_consumption(t):
    ramp_up = 1 / (1 + np.exp(-2 * (t - 8)))
    ramp_down = 1 / (1 + np.exp(2 * (t - 17)))
    return 0.2 + 0.8 * ramp_up * ramp_down

# Civic buildings:
def hospital_consumption(t):
    return 1.0 + 0.1 * np.sin(2 * np.pi * (t - 6) / 24)

def library_consumption(t):
    baseline = 0.2
    peak = 1.0 * np.exp(-((t - 14) / 1.0) ** 2)
    return baseline + peak

def school_consumption(t):
    baseline = 0.2
    morning_peak = 1.2 * np.exp(-((t - 8) / 1.0) ** 2)
    afternoon_peak = 1.0 * np.exp(-((t - 15) / 1.0) ** 2)
    return baseline + morning_peak + afternoon_peak

# Industrial types:
import numpy as np

# Factory: Instead of a sinusoid, this uses a high constant level with a small midday hump.
def factory_consumption(t):
    # Base consumption is high during operating hours.
    # A modest Gaussian hump is added around noon (t = 12) to simulate increased activity.
    base = 1.8
    midday_hump = 0.2 * np.exp(-((t - 12) / 2)**2)
    return base + midday_hump

# Warehouse: Use piecewise linear interpolation to simulate a clear shift in usage.
def warehouse_consumption(t):
    # Define key times and corresponding consumption levels:
    # Very low before 8:00, a plateau from 8:00 to 18:00, and then low again.
    key_times = np.array([0, 8, 18, 24])
    # These values create a trapezoidal profile.
    key_values = np.array([0.5, 1.5, 1.5, 0.5])
    return np.interp(t, key_times, key_values)

# Processing Plant: Mostly constant consumption with a short dip mid-day.
def processing_plant_consumption(t):
    # Base consumption remains nearly constant.
    # A small dip is introduced around 13:00 (1 PM) to simulate a brief slowdown (e.g., a scheduled break).
    base = 1.2
    midday_dip = -0.2 * np.exp(-((t - 13) / 1)**2)
    return base + midday_dip


# --- Map each building type to its consumption function ---
consumption_functions = {
    "single_family": single_family_consumption,
    "apartment": apartment_consumption,
    "restaurant": restaurant_consumption,
    "office": office_consumption,
    "hospital": hospital_consumption,
    "library": library_consumption,
    "school": school_consumption,
    "factory": factory_consumption,
    "warehouse": warehouse_consumption,
    "processing_plant": processing_plant_consumption
}

num_types = len(consumption_functions)
cols = 3
rows = int(np.ceil(num_types / cols))

fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=False)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.3)

axs = axs.flatten()

for ax, (btype, func) in zip(axs, consumption_functions.items()):
    consumption = func(t)
    ax.plot(t, consumption, label=btype)
    # Set title with padding
    #ax.set_title(btype.replace("_", " ").title(), pad=10)
    # Ensure integer hours from 0 to 24
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlim(0, 24)
    ax.set_ylabel("Consumption (units)", labelpad=5)
    ax.grid(True)
    ax.legend()

# Remove any unused subplots if we don't have enough building types to fill them
for ax in axs[num_types:]:
    ax.remove()

plt.tight_layout()
plt.show()
