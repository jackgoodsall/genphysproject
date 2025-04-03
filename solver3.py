import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.constants import g
import pandas as pd
import os

os.makedirs('trajectory_data', exist_ok=True)

class ODESystem:
    def __init__(self, drag_coefficient, x0, spring_constant, launch_angle_deg, y0=5364):
        self.mass = 85  # kg
        self.radius = 0.2  # m
        self.area = np.pi * (self.radius ** 2)
        self.x0 = x0
        self.y0 = y0
        self.spring_constant = spring_constant
        self.launch_angle = np.deg2rad(launch_angle_deg)
        self.mountain_height = 8848
        self.drag_coefficient = drag_coefficient

        v0 = np.sqrt(spring_constant / self.mass)
        self.Vx0 = v0 * np.cos(self.launch_angle)
        self.Vy0 = v0 * np.sin(self.launch_angle)

    def system(self, t, z):
        x, y, Vx, Vy = z
        rho = 1.293 * np.exp(-y / self.mountain_height)
        V = np.hypot(Vx, Vy)
        drag = 0.5 * rho * self.area * self.drag_coefficient * V
        
        return [
            Vx,
            Vy,
            -drag * Vx / self.mass,
            -g - drag * Vy / self.mass
        ]

    def solve(self, t_max=10000):
        z0 = [self.x0, self.y0, self.Vx0, self.Vy0]

        return solve_ivp(
            self.system,
            (0, t_max),
            z0,
            method='LSODA',
            events=(self.target_event, self.ground_event),
            dense_output=True,
            rtol=1e-4,
            atol=1e-7,
            max_step=0.1
        )

    def target_event(self, t, z):
        x, y = z[0], z[1]
        return max(abs(x) - 5, abs(y - 8848) - 10)
    target_event.terminal = True
    
    def ground_event(self, t, z):
        return z[1]
    ground_event.terminal = True
    ground_event.direction = -1

successful_trajectories = pd.DataFrame(columns=['spring_constant', 'launch_angle', 'x0', 'time_to_target', 'final_vy'])

def objective(params):
    log_k, angle_deg, x0_mag = params
    spring = 10 ** log_k
    x0 = -x0_mag
    
    try:
        system = ODESystem(0.3, x0, spring, angle_deg)
        sol = system.solve(300)
        
        if sol.t_events[0].size > 0:  # Target area reached
            # Get precise state at event time
            t_event = sol.t_events[0][0]
            vy_at_target = sol.sol(t_event)[3]
            
            # Require negative vertical velocity (descending)
            if vy_at_target < 0:
                successful_trajectories.loc[len(successful_trajectories)] = [
                    spring, angle_deg, x0, t_event, vy_at_target
                ]
                return t_event
    except:
        pass
    
    return np.inf  # Penalty for failed attempts

# Optimization bounds
bounds = [
    (6, 9),        # log10(spring_constant) range: 1e6 to 1e9 N/m
    (30, 70),      # Launch angle in degrees
    (2500, 7500)   # Starting distance magnitude
]

result = differential_evolution(
    objective,
    bounds,
    strategy='best1exp',
    popsize=10,
    mutation=(0.6, 1.2),
    recombination=0.7,
    maxiter=1000,
    seed=42,
    disp=True
)

# Extract optimal parameters
optimal_log_k, optimal_angle, optimal_x0_mag = result.x
optimal_spring = 10 ** optimal_log_k
optimal_x0 = -optimal_x0_mag

print("\nOptimization Results:")
print(f"Spring Constant: {optimal_spring:.2e} N/m")
print(f"Launch Angle: {optimal_angle:.2f}°")
print(f"Start Position: {optimal_x0:.1f} m")
print(f"Minimum Time: {result.fun:.2f} s")

# Save results
successful_trajectories.to_csv('trajectory_data/optimized_results.csv', index=False)

# Plot best trajectory with velocity check
if not successful_trajectories.empty:
    best = successful_trajectories.loc[successful_trajectories['time_to_target'].idxmin()]
    system = ODESystem(0.7, best['x0'], best['spring_constant'], best['launch_angle'])
    sol = system.solve()
    
    # Get detailed trajectory
    t_plot = np.linspace(0, sol.t[-1], 100000)
    x_plot, y_plot, vx_plot, vy_plot = sol.sol(t_plot)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Trajectory plot
    ax1.plot(x_plot, y_plot, color='navy')
    ax1.scatter(0, 8848, c='red', s=100, label='Target')
    ax1.scatter(best['x0'], 5364, marker='*', c='lime', s=200, label='Launch')
    ax1.set_ylabel('Altitude (m)')
    ax1.legend()
    ax1.grid(True)
    
    # Velocity plot
    ax2.plot(t_plot, vy_plot, color='maroon', label='Vertical Velocity (Vy)')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Vy (m/s)')
    ax2.legend()
    ax2.grid(True)
    
    # Mark target arrival
    arrival_time = best['time_to_target']
    ax1.axvline(0, color='red', linestyle=':', alpha=0.5)
    ax2.axvline(arrival_time, color='red', linestyle=':', label='Target Impact')
    
    plt.suptitle(f'Optimal Trajectory with Descending Impact\nVy at target: {best["final_vy"]:.2f} m/s')
    plt.tight_layout()
    plt.savefig('trajectory_data/velocity_verified_trajectory.png', dpi=300)
    plt.show()