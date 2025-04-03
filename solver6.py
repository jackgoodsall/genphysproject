import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.constants import g
import pandas as pd
import os
import random
from datetime import datetime

# Create data directories
os.makedirs('trajectory_data/successful', exist_ok=True)
os.makedirs('trajectory_data/failed', exist_ok=True)

class ODESystem:
    def __init__(self, drag_coefficient, x0, spring_constant, launch_angle_deg, y0=5364):
        self.mass = 85  # kg
        self.radius = 0.2  # m
        self.area = np.pi * (self.radius ** 2)
        self.drag_coefficient = drag_coefficient
        self.mountain_height = 8848  # m (Everest height)
        self.x0 = x0
        self.y0 = y0
        self.spring_constant = spring_constant
        self.launch_angle = np.deg2rad(launch_angle_deg)
        
        v0 = np.sqrt(spring_constant / self.mass)
        self.Vx0 = v0 * np.cos(self.launch_angle)
        self.Vy0 = v0 * np.sin(self.launch_angle)

    def system(self, t, z):
        x, y, Vx, Vy = z
        rho = 1.293 * np.exp(-y / self.mountain_height)
        speed = np.sqrt(Vx**2 + Vy**2)
        drag = 0.5 * rho * self.area * self.drag_coefficient * speed
        return [Vx, Vy, -drag*Vx/self.mass, -g - drag*Vy/self.mass]

    def solve(self, t_max=1000, low_precision=False):
        solver_params = {
            'method': 'LSODA',
            'events': (self.target_event, self.ground_event),
            'dense_output': True,
            'rtol': 1e-3 if low_precision else 1e-6,
            'atol': 1e-4 if low_precision else 1e-9,
            'max_step': 1.0 if low_precision else 0.001
        }
        return solve_ivp(self.system, (0, t_max), [self.x0, self.y0, self.Vx0, self.Vy0], **solver_params)

    def target_event(self, t, z):
        x, y = z[0], z[1]
        return max(abs(x) - 5, abs(y - 8848) - 10)
    target_event.terminal = True
    target_event.direction = -1

    def ground_event(self, t, z):
        return z[1] - 5364
    ground_event.terminal = True
    ground_event.direction = -1

# Data storage
successful_trajs = pd.DataFrame(columns=[
    'timestamp', 'spring', 'angle', 'x0', 'impact_time', 
    'final_vx', 'final_vy', 'impact_x', 'impact_y'
])

failed_trajs = pd.DataFrame(columns=[
    'timestamp', 'spring', 'angle', 'x0', 
    'reason', 'final_x', 'final_y', 'final_vx', 'final_vy'
])

def save_trajectories():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not successful_trajs.empty:
        successful_trajs.to_csv(f'trajectory_data/successful/success_{timestamp}.csv', index=False)
    if not failed_trajs.empty:
        failed_trajs.to_csv(f'trajectory_data/failed/failed_{timestamp}.csv', index=False)

def log_failure(params, sol, reason):
    if random.random() > 0.0001: return
    
    entry = {
        'timestamp': datetime.now(),
        'spring': 10**params[0],
        'angle': params[1],
        'x0': -params[2],
        'reason': reason,
        'final_x': sol.y[0,-1] if sol else np.nan,
        'final_y': sol.y[1,-1] if sol else np.nan,
        'final_vx': sol.y[2,-1] if sol else np.nan,
        'final_vy': sol.y[3,-1] if sol else np.nan
    }
    global failed_trajs
    failed_trajs = pd.concat([failed_trajs, pd.DataFrame([entry])], ignore_index=True)

def objective(params):
    global successful_trajs
    spring = 10**params[0]
    angle = params[1]
    x0 = -params[2]
    
    try:
        system = ODESystem(0.7, x0, spring, angle)
        sol = system.solve(300, low_precision=True)
        
        if sol.t_events[0].size > 0:
            t_impact = sol.t_events[0][0]
            x, y, vx, vy = sol.sol(t_impact)
            if abs(x) <= 5 and abs(y-8848) <= 10 and vy < 0:
                entry = {
                    'timestamp': datetime.now(),
                    'spring': spring,
                    'angle': angle,
                    'x0': x0,
                    'impact_time': t_impact,
                    'final_vx': vx,
                    'final_vy': vy,
                    'impact_x': x,
                    'impact_y': y
                }
                successful_trajs = pd.concat([successful_trajs, pd.DataFrame([entry])], ignore_index=True)
                return t_impact
            else:
                log_failure(params, sol, 'target_miss')
        else:
            log_failure(params, sol, 'no_impact')
    except Exception as e:
        log_failure(params, None, f'error_{str(e)[:50]}')
    return np.inf

def plot_comparison():
    # Load saved data
    
    success_files = ["success_20250402_230438.csv"]
    failed_files = ["failed_20250402_230438.csv"]
    if not success_files or not failed_files:
        print("No saved trajectories found!")
        return

    # Load and process data
    success_df = pd.concat([pd.read_csv(f'trajectory_data/successful/{f}') for f in success_files])
    optimal = success_df.loc[success_df.impact_time.idxmin()]
    unoptimal = success_df.loc[success_df.impact_time.idxmax()]
    failed_df = pd.concat([pd.read_csv(f'trajectory_data/failed/{f}') for f in failed_files])
    failed = failed_df.sample(1).iloc[0]

    # Create figure with extended x-axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get axis limits that will be used for the plot
    x_min, x_max = -4000, 2000  # Extended x-axis range
    y_min, y_max = 0, 10000
    
    # Mountain profile with extended base
    def mountain_profile(y0=5364):
        peak_width = 5
        return np.array([
            [-1500, y0],          # Far left base
            [-peak_width/2, 8848], # Left slope
            [peak_width/2, 8848],  # Right slope
            [1500, y0],          # Far right base         # Close polygon
        ])
    y0=5364
    # Create full terrain base (fills entire plot bottom)
    terrain_base = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y0],
        [x_min, y0],
        [x_min, y_min]
    ])
    
    # Plot terrain and mountain
    ax.fill(terrain_base[:,0], terrain_base[:,1], color='#8B4513', alpha=0.4, zorder=1)
    mountain = mountain_profile()
    ax.fill(mountain[:,0], mountain[:,1], color='#8B4513', alpha=0.4,  zorder=2)
    
    # Plot trajectories
    for case, color, style, label in [
        (optimal, 'limegreen', '-', 'Optimal'),
        #(unoptimal, 'orange', '-', 'Unoptimal'), 
        #(failed, 'red', '--', 'Failed')
    ]:
        try:
            system = ODESystem(0.7, case.x0, case.spring, case.angle)
            sol = system.solve()
            t = np.linspace(0, sol.t[-1], 100000)
            x, y, _, _ = sol.sol(t)
            ax.plot(x, y, color=color, linestyle=style, label=label, lw=1.5, zorder=3)
            ax.scatter(case.x0, 5364, color=color, s=100, edgecolor='k', zorder=4)
        except Exception as e:
            print(f"Error plotting {label}: {str(e)}")

    # Formatting
    #ax.scatter([0], [8848], c='gold', s=200, edgecolor='k', marker='*', label='Target', zorder=5)
    ax.set_xlabel("Horizontal distance from Mount Everest peak (m) ")
    ax.set_ylabel("Elevation above sea level (m)")

    ax.grid(True, alpha=0.4)
    ax.legend()
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('trajectory_data/comparison_plot.png', dpi=300)
    plt.show()

# Optimization setup
bounds = [(6, 9), (30, 70), (2500, 7500)]
# result = differential_evolution(
#     objective, bounds,
#     strategy='best1exp',
#     popsize=10,
#     mutation=(0.5, 1.5),
#     recombination=0.75,
#     maxiter=250,
#     seed=42,
#     disp=True
# )

# Save and plot results
save_trajectories()
plot_comparison()

print("\nOptimization complete!")
print(f"Best impact time: {result.fun:.2f}s")
print(f"Parameters: Spring={10**result.x[0]:.2e} N/m, Angle={result.x[1]:.1f}°, X0={-result.x[2]:.0f}m")