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
        # System parameters
        self.mass = 85  # kg
        self.radius = 0.2  # m
        self.area = np.pi * (self.radius ** 2)
        self.drag_coefficient = drag_coefficient
        self.mountain_height = 8848  # m (Everest height)
        
        # Initial conditions
        self.x0 = x0
        self.y0 = y0
        self.spring_constant = spring_constant
        self.launch_angle = np.deg2rad(launch_angle_deg)
        
        # Calculate initial velocity
        v0 = np.sqrt(spring_constant / self.mass)
        self.Vx0 = v0 * np.cos(self.launch_angle)
        self.Vy0 = v0 * np.sin(self.launch_angle)

    def system(self, t, z):
        x, y, Vx, Vy = z
        # Air density decreases with altitude
        rho = 1.293 * np.exp(-y / self.mountain_height)
        speed = np.sqrt(Vx**2 + Vy**2)
        drag = 0.5 * rho * self.area * self.drag_coefficient * speed
        
        return [
            Vx,  # dx/dt
            Vy,  # dy/dt
            -drag * Vx / self.mass,  # dVx/dt
            -g - drag * Vy / self.mass  # dVy/dt
        ]

    def solve(self, t_max=1000, low_precision=False):
        solver_params = {
            'method': 'RK45',
            'events': (self.target_event, self.ground_event),
            'dense_output': True,
            'rtol': 1e-6 if low_precision else 1e-6,
            'atol': 1e-9 if low_precision else 1e-9,
            'max_step': 0.2 if low_precision else 0.001
        }
        
        return solve_ivp(
            self.system,
            (0, t_max),
            [self.x0, self.y0, self.Vx0, self.Vy0],
            **solver_params
        )

    def target_event(self, t, z):
        x, y = z[0], z[1]
        return max(abs(x) - 2.5, abs(y - 8848) - 2)
    target_event.terminal = True
    target_event.direction = -1  # Only trigger when entering target

    def ground_event(self, t, z):
        return z[1] - 5364  # Detect ground impact
    ground_event.terminal = True
    ground_event.direction = -1

# Data collection setup
successful_trajs = pd.DataFrame(columns=[
    'timestamp', 'spring', 'angle', 'x0',
    'impact_time', 'final_vx', 'final_vy',
    'impact_x', 'impact_y'
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
    if random.random() > 0.0001:  # 0.01% chance to log
        return
    
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
        
        if sol.t_events[0].size > 0:  # Target hit
            t_impact = sol.t_events[0][0]
            x, y, vx, vy = sol.sol(t_impact)
            
            if abs(x) <= 2.5 and abs(y-8848) <= 2 and vy < 0:
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

# Optimization setup
bounds = [
    (6, 9),        # log10(spring) [1e6-1e9 N/m]
    (20, 80),      # angle [degrees]
    (2500, 7500)   # x0 magnitude [m]
]

result = differential_evolution(
    objective,
    bounds,
    strategy='best1exp',
    popsize=15,
    mutation=(0.6, 1.5),
    recombination=0.75,
    maxiter=1000,
    seed=42,
    disp=True,
    polish=False
)

# Final save and validation
save_trajectories()
if not successful_trajs.empty:
    # Validate best result
    best = successful_trajs.loc[successful_trajs.impact_time.idxmin()]
    system = ODESystem(0.7, best.x0, best.spring, best.angle)
    val_sol = system.solve(low_precision=False)
    
    # Create trapezoidal mountain model
    def mountain_profile():
        peak_width = 10  # Width at summit
        base_width = 1500  # Width at base
        return np.array([
            [-base_width/2, 0],            # Left base
            [-peak_width/2, 8848],         # Left slope
            [peak_width/2, 8848],          # Right slope
            [base_width/2, 0],             # Right base
            [-base_width/2, 0]             # Close polygon
        ])

    # High-res plotting
    if val_sol.t_events[0].size > 0:
        t_plot = np.linspace(0, val_sol.t[-1], 1000)
        x, y, _, _ = val_sol.sol(t_plot)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot mountain
        mountain = mountain_profile()
        ax.fill(mountain[:,0], mountain[:,1], color='#8B4513', 
                alpha=0.4, label='Mountain Profile', zorder=1)
        
        # Plot trajectory
        ax.plot(x, y, 'b-', lw=1.5, label='Projectile Path', zorder=3)
        ax.scatter([0, best.x0], [8848, 5364], 
                   c=['red', 'limegreen'], s=100, 
                   edgecolors='k', zorder=4,
                   label=['Target', 'Launch Point'])
        
        # Formatting
        ax.set_xlabel('Horizontal Distance (m)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title("Projectile Trajectory with Mountain Profile")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Set dynamic axis limits
        pad = 500
        ax.set_xlim(min(x.min(), mountain[:,0].min())-pad,
                    max(x.max(), mountain[:,0].max())+pad)
        ax.set_ylim(0, max(y.max(), 8848)+1000)
        
        plt.tight_layout()
        plt.savefig('trajectory_data/trajectory_with_mountain.png', dpi=300)
        plt.show()
print("\nOptimization complete!")
print(f"Best impact time: {result.fun:.2f}s")
print(f"Parameters: Spring={10**result.x[0]:.2e} N/m, Angle={result.x[1]:.1f}°, X0={-result.x[2]:.0f}m")