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
            rtol=1e-6,  # Tighter tolerances
            atol=1e-9,
            max_step=0.002
        )

    def target_event(self, t, z):
        x, y = z[0], z[1]
        return max(abs(x) - 5, abs(y - 8848) - 10)
    target_event.terminal = True
    target_event.direction = -1  # Only trigger when entering target area

    def ground_event(self, t, z):
        return z[1]
    ground_event.terminal = True
    ground_event.direction = -1

successful_trajectories = pd.DataFrame(columns=[
    'spring_constant', 'launch_angle', 'x0', 
    'time_to_target', 'final_vx', 'final_vy',
    'target_x', 'target_y'
])

def objective(params):
    log_k, angle_deg, x0_mag = params
    spring = 10 ** log_k
    x0 = -x0_mag
    
    try:
        system = ODESystem(0.3, x0, spring, angle_deg)
        sol = system.solve(300)
        
        # Check for successful target impact
        if sol.t_events[0].size > 0:  # Target event triggered
            t_event = sol.t_events[0][0]
            x, y, vx, vy = sol.sol(t_event)
            
            # Strict success criteria
            in_target_x = abs(x) <= 5
            in_target_y = abs(y - 8848) <= 10
            descending = vy < 0
            
            if in_target_x and in_target_y and descending:
                successful_trajectories.loc[len(successful_trajectories)] = {
                    'spring_constant': spring,
                    'launch_angle': angle_deg,
                    'x0': x0,
                    'time_to_target': t_event,
                    'final_vx': vx,
                    'final_vy': vy,
                    'target_x': x,
                    'target_y': y
                }
                return t_event
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return np.inf
    
    return np.inf  # Penalize non-successful attempts

# Optimization bounds
bounds = [
    (6, 9),    # log10(spring_constant): 1e6 to 1e8 N/m
    (30, 70),      # Launch angle in degrees
    (2500, 7500)   # Starting distance magnitude
]

result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    popsize=12,
    mutation=(0.5, 1.5),
    recombination=0.7,
    maxiter=150,
    seed=42,
    disp=True,
    polish=True
)

# Post-optimization validation
if result.success:
    optimal_params = {
        'spring': 10 ** result.x[0],
        'angle': result.x[1],
        'x0': -result.x[2]
    }
    
    # Validate with high precision
    validation_system = ODESystem(
        drag_coefficient=0.3,
        x0=optimal_params['x0'],
        spring_constant=optimal_params['spring'],
        launch_angle_deg=optimal_params['angle']
    )
    
    val_sol = validation_system.solve()
    
    if val_sol.t_events[0].size > 0:
        t_val = val_sol.t_events[0][0]
        x_val, y_val, vx_val, vy_val = val_sol.sol(t_val)
        
        valid = (
            abs(x_val) <= 5 and 
            abs(y_val - 8848) <= 10 and 
            vy_val < 0
        )
        
        if not valid:
            print("\nOptimization result failed final validation!")
            successful_trajectories = successful_trajectories.drop(
                successful_trajectories.index[-1]
            )

# Save and plot results
if not successful_trajectories.empty:
    successful_trajectories.to_csv(
        'trajectory_data/validated_solutions.csv', 
        index=False
    )
    
    best = successful_trajectories.loc[successful_trajectories['time_to_target'].idxmin()]
    system = ODESystem(0.3, best['x0'], best['spring_constant'], best['launch_angle'])
    sol = system.solve()
    
    # High-resolution plotting
    t_plot = np.linspace(0, sol.t[-1], 10000)
    x_plot, y_plot, vx_plot, vy_plot = sol.sol(t_plot)
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Trajectory plot
    ax[0].plot(x_plot, y_plot, color='darkblue')
    ax[0].scatter(0, 8848, c='red', s=100, label='Target')
    ax[0].scatter(best['x0'], 5364, marker='*', s=200, c='lime', label='Launch')
    ax[0].set_ylabel('Altitude (m)')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()
    
    # Velocity plot
    ax[1].plot(t_plot, vy_plot, color='maroon')
    ax[1].axhline(0, color='gray', linestyle='--')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Vertical Velocity (m/s)')
    ax[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Validated Optimal Trajectory\nImpact Time: {best['time_to_target']:.2f}s")
    plt.tight_layout()
    plt.savefig('trajectory_data/final_validated_trajectory.png', dpi=300)
    plt.show()
else:
    print("No valid trajectories found!")