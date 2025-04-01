import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.constants import g
import pandas as pd
import os

# Create a directory to store results if it doesn't exist
os.makedirs('trajectory_data', exist_ok=True)

class ODESystem:
    def __init__(self, drag_coefficient, x0, spring_constant, launch_angle_deg, y0=5_364):
        """
        Initialize the ODE system with parameters and initial conditions.
        """
        self.mass = 85  # kg
        self.radius = 0.2  # m
        self.area = np.pi * (self.radius ** 2)  # m^2
        self.x0 = x0  # Initial horizontal position
        self.y0 = y0  # Initial vertical position
        self.spring_constant = spring_constant  # N/m
        self.launch_angle = np.deg2rad(launch_angle_deg)
        self.mountain_height = 8_848  # m
        self.air_density = 1.293  # kg/m^3
        self.drag_coefficient = drag_coefficient

        # Compute initial speed from spring energy:
        v0 = np.sqrt(self.spring_constant / self.mass)
        self.Vx0 = v0 * np.cos(self.launch_angle)
        self.Vy0 = v0 * np.sin(self.launch_angle)

    def system(self, t, z):
        """
        Defines the system of ODEs.
        """
        x, y, V_x, V_y = z
        dxdt = V_x
        dydt = V_y
        V = np.sqrt(V_x**2 + V_y**2)
        dVxdt = -self.area * V_x * np.exp(-y / self.mountain_height) * V / (2 * self.mass) * self.drag_coefficient
        dVydt = -g - (V_y * V * self.area * self.drag_coefficient) / (2 * self.mass)
        return [dxdt, dydt, dVxdt, dVydt]

    def solve(self, t_max=100000):
        """
        Solves the ODE system using solve_ivp with event handling.
        """
        z0 = [self.x0, self.y0, self.Vx0, self.Vy0]
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 100000000)
        solution = solve_ivp(
            self.system, t_span, z0,
            events=[event_target_area, event_vx_near_zero, event_y_cross_zero],
            max_step=0.002, t_eval= t_eval
        )
        return solution

def event_target_area(t, z):
    x_target = 0
    y_target = 8_848  # Mount Everest height in meters
    tolerance = 20   # meters
    x, y = z[0], z[1]
    return max(abs(x - x_target) - tolerance, abs(y - y_target) - tolerance)
event_target_area.terminal = True

def event_vx_near_zero(t, z):
    return z[2] if abs(z[2]) > 0.001 else 0
event_vx_near_zero.terminal = True

def event_y_cross_zero(t, z):
    return z[1]
event_y_cross_zero.terminal = True
event_y_cross_zero.direction = -1

# Initialize DataFrame to store results
successful_trajectories = pd.DataFrame(columns=['spring_constant', 'launch_angle', 'time_to_target'])

def objective(params):
    global successful_trajectories
    
    # Convert parameters from log scale for spring constant
    log_spring_constant, launch_angle_deg = params
    spring_constant = 10 ** log_spring_constant
    
    try:
        ode_solver = ODESystem(drag_coefficient=0.3, x0=-4000, 
                             spring_constant=spring_constant, 
                             launch_angle_deg=launch_angle_deg)
        solution = ode_solver.solve()
        
        if len(solution.t_events[0]) > 0:  # Check target area event
            time_to_target = solution.t_events[0][0]
            new_entry = pd.DataFrame({
                'spring_constant': [spring_constant],
                'launch_angle': [launch_angle_deg],
                'time_to_target': [time_to_target]
            })
            successful_trajectories = pd.concat([successful_trajectories, new_entry], ignore_index=True)
            return time_to_target
    except:
        pass
    
    return np.inf  # Use infinity for failed cases

# Set up bounds in log scale for spring constant
bounds = [
    (6, 10),        # log10(spring_constant) range: 1e6 to 1e16 N/m
    (10, 80)        # Launch angle in degrees
]

# Run differential evolution
result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=200,
    popsize=15,
    mutation=(0.5, 1),
    recombination=0.6,
    seed=42,
    disp=True, # Use all available cores
)

# Convert optimal parameters back from log scale
optimal_log_k, optimal_angle = result.x
optimal_spring_constant = 10 ** optimal_log_k

print(f"\nOptimization Results:")
print(f"Optimal Spring Constant: {optimal_spring_constant:.2e} N/m")
print(f"Optimal Launch Angle: {optimal_angle:.2f} degrees")
print(f"Minimum Time to Target: {result.fun:.2f} seconds")

# Save results
successful_trajectories.to_csv('trajectory_data/de_successful_trajectories.csv', index=False)
print(f"\nSaved {len(successful_trajectories)} successful trajectories")

# Plot best trajectory
if len(successful_trajectories) > 0:
    best = successful_trajectories.loc[successful_trajectories['time_to_target'].idxmin()]
    ode_solver = ODESystem(drag_coefficient=0.3, x0=-2500,
                          spring_constant=best['spring_constant'],
                          launch_angle_deg=best['launch_angle'])
    solution = ode_solver.solve()
    
    plt.figure(figsize=(12, 6))
    plt.plot(solution.y[0], solution.y[1], label='Trajectory')
    plt.scatter(0, 8848, c='red', s=100, label='Target (Mt. Everest)')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Altitude (m)')
    plt.title(f'Optimal Trajectory\nSpring Constant: {best["spring_constant"]:.2e} N/m | Angle: {best["launch_angle"]:.1f}°')
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectory_data/de_optimal_trajectory.png')
    plt.show()