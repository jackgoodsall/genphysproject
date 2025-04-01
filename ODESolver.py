import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
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

    def solve(self, t_max=10000000):
        """
        Solves the ODE system using solve_ivp with event handling.
        """
        z0 = [self.x0, self.y0, self.Vx0, self.Vy0]
        t_span = (0, t_max)
        time_a = np.linspace(0, t_max, 100000000)
        solution = solve_ivp(
            self.system, t_span, z0,
            events=[event_target_area, event_vx_near_zero, event_y_cross_zero],
            t_eval=time_a
        )
        return solution

def event_target_area(t, z):
    """
    Stops integration when x is within 10 meters of 0 and y is within 10 meters of mountain peak height.
    """
    x_target = 0
    y_target = 8_848  # Mount Everest height in meters
    tolerance = 10   # meters

    x, y = z[0], z[1]
    x_diff = abs(x - x_target)
    y_diff = abs(y - y_target)

    # Event occurs when both x and y are within the specified tolerance
    return max(x_diff - tolerance, y_diff - tolerance)

event_target_area.terminal = True  
event_target_area.direction = 0  

def event_vx_near_zero(t, z):
    """
    Stops integration when V_x is close to 0 (within ±1 m/s).
    """
    V_x = z[2]  
    if -0.001 <= V_x <= 0.001:  
        return 0  
    return V_x  

event_vx_near_zero.terminal = True  
event_vx_near_zero.direction = 0  

def event_y_cross_zero(t, z):
    """
    Stops integration when y crosses zero (only when decreasing).
    """
    y = z[1]
    return y

event_y_cross_zero.terminal = True  
event_y_cross_zero.direction = -1  # Only trigger when y decreases through zero

# Initialize a DataFrame to store successful trajectories
successful_trajectories = pd.DataFrame(columns=['spring_constant', 'launch_angle', 'time_to_target'])

def objective(params):
    """
    Objective function to minimize: time to reach the target area.
    Only considers simulations that ended due to the event_target_area event.
    """
    global successful_trajectories
    
    spring_constant, launch_angle_deg = params
    ode_solver = ODESystem(drag_coefficient=0.5, x0=-2000, 
                          spring_constant=spring_constant, 
                          launch_angle_deg=launch_angle_deg)
    solution = ode_solver.solve()
    
    # Check if the simulation ended due to the event_target_area event
    if len(solution.t_events[0]) > 0:
        time_to_target = solution.t_events[0][0]
        # Save successful trajectory parameters
        new_entry = pd.DataFrame({
            'spring_constant': [spring_constant],
            'launch_angle': [launch_angle_deg],
            'time_to_target': [time_to_target]
        })
        successful_trajectories = pd.concat([successful_trajectories, new_entry], ignore_index=True)
        return time_to_target
    else:
        return 1e20  # Large penalty for unsuccessful trajectories

# Initial guess
initial_guess = [5e9, 45]  # spring_constant = 5e9 N/m, launch_angle_deg = 45 degrees

# Bounds for the parameters
bounds = [(1e6, 1e16), (10, 80)]  # Wide bounds for exploration

# Perform the optimization
result = minimize(objective, initial_guess, bounds=bounds, method='Powell')

# Output the results
optimal_spring_constant, optimal_launch_angle_deg = result.x
minimum_time = result.fun

print(f"Optimal Spring Constant: {optimal_spring_constant:.2e} N/m")
print(f"Optimal Launch Angle: {optimal_launch_angle_deg:.2f} degrees")
print(f"Minimum Time to Target: {minimum_time:.2f} seconds")

# Save all successful trajectories to CSV
successful_trajectories.to_csv('trajectory_data/successful_trajectories.csv', index=False)
print(f"\nSaved {len(successful_trajectories)} successful trajectories to 'trajectory_data/successful_trajectories.csv'")

# Plot some successful trajectories if desired
if len(successful_trajectories) > 0:
    # Plot the best trajectory
    best_params = successful_trajectories.loc[successful_trajectories['time_to_target'].idxmin()]
    ode_solver = ODESystem(drag_coefficient=0.3, x0=-2000,
                          spring_constant=best_params['spring_constant'],
                          launch_angle_deg=best_params['launch_angle'])
    solution = ode_solver.solve()
    
    plt.figure(figsize=(10, 6))
    plt.plot(solution.y[0], solution.y[1])
    plt.scatter(0, 8848, color='red', label='Target (Mt. Everest)')
    plt.xlabel('Horizontal Position (m)')
    plt.ylabel('Vertical Position (m)')
    plt.title(f'Optimal Trajectory\nSpring Constant: {best_params["spring_constant"]:.2e} N/m, Angle: {best_params["launch_angle"]:.1f}°')
    plt.legend()
    plt.grid()
    plt.savefig('trajectory_data/optimal_trajectory.png')
    plt.show()


from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Define the search space with log-transformed spring constant
space = [
    Real(6, 16, name='log_spring_constant', prior='uniform'),  # log10 scale
    Integer(20, 70, name='launch_angle')
]

@use_named_args(space)
def objective_skopt(log_spring_constant, launch_angle):
    spring_constant = 10 ** log_spring_constant
    return objective([spring_constant, launch_angle])

result = gp_minimize(
    objective_skopt, space, n_calls=50, 
    acq_func='EI', random_state=42, verbose=True
)