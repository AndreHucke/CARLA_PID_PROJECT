import carla
import time
import math
import numpy as np
from pid_controller import PIDController
from route_manager import RouteManager
import os
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    'TOWN': 'Town02',
    'SPAWN_INDEX': 73,
    'SPEED_KMH': 25,
    'WAYPOINTS_BASE_PATH': "CARLA_PID_PROJECT/Main",
    'SCENARIOS': ['waypoints0.xml', 'waypoints1.xml',
                  'waypoints2.xml', 'waypoints3.xml',
                  'waypoints4.xml', 'waypoints5.xml',
                  'waypoints6.xml', 'waypoints7.xml',
                  'waypoints8.xml', 'waypoints9.xml'],
    'CONTROL_RATE': 0.1,
    'SLEEP_TIME': 0.01,
    'PID': {
        'throttle': {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.0},
        'steering': {'Kp': 0.5, 'Ki': 0.0, 'Kd': 0.1}
    },
    'SPEED_THRESHOLD': 0.5,
    'CRASH_IMPULSE_THRESHOLD': 50.0,
    'PATH_PLANNING': {
        'LOOKAHEAD_POINTS': 3,
        'WEIGHTS': [0.6, 0.3, 0.1],
        'MIN_DISTANCE': 2.0,
        'PREDICTION_TIME': 0.5
    },
}

class SpawnCar:
    def __init__(self, world, blueprint_library, spawn_point, route_manager):
        self.world = world
        self.actor_list = []
        self._setup_vehicle(blueprint_library, spawn_point)
        self._setup_controllers()
        self._setup_collision_sensor(blueprint_library)
        self.route_manager = route_manager
        self.current_target = None
        self.collision_detected = False
        self.last_collision_impulse = 0.0
        self.next_waypoint_distance = 0.0
        self.speed_history = []
        self.time_history = []
        self.start_time = time.time()
        self.future_waypoints = []
        self.lateral_error_history = []

    def _setup_vehicle(self, blueprint_library, spawn_point):
        """Setup the vehicle actor"""
        try:
            self.vehicle_bp = blueprint_library.filter('mini')[0]
            self.vehicle = self.world.spawn_actor(self.vehicle_bp, spawn_point)
            self.actor_list.append(self.vehicle)
        except Exception as e:
            raise RuntimeError(f"Failed to spawn vehicle: {e}")

    def _setup_controllers(self):
        """Initialize PID controllers"""
        pid = CONFIG['PID']
        self.throttle_controller = PIDController(**pid['throttle'])
        self.steering_controller = PIDController(**pid['steering'])
        self.target_speed = CONFIG['SPEED_KMH']

    def _setup_collision_sensor(self, blueprint_library):
        """Setup collision sensor"""
        try:
            collision_bp = blueprint_library.find('sensor.other.collision')
            collision_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
            self.collision_sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.vehicle)
            self.collision_sensor.listen(self._on_collision)
            self.actor_list.append(self.collision_sensor)
        except Exception as e:
            print(f"Warning: Failed to setup collision sensor: {e}")

    def _on_collision(self, event):
        """Callback for collision events"""
        impulse = math.sqrt(
            event.normal_impulse.x**2 +
            event.normal_impulse.y**2 +
            event.normal_impulse.z**2
        )
        self.last_collision_impulse = impulse
        if impulse > CONFIG['CRASH_IMPULSE_THRESHOLD']:
            self.collision_detected = True
            print(f"\nCRASH DETECTED! Collision force: {impulse:.1f}")

    def get_speed(self):
        velocity = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def update_control(self, dt):
        """Update vehicle control based on current state and target"""
        try:
            vehicle_transform = self.vehicle.get_transform()
            current_speed = self.get_speed()
            
            # Record speed data
            current_time = time.time() - self.start_time
            self.speed_history.append(current_speed)
            self.time_history.append(current_time)

            if self.collision_detected:
                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True
                ))
                return False

            # Check if we need a new target
            if self._need_new_target(vehicle_transform):
                if not self._update_target():
                    self.vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0,
                        brake=1.0,
                        hand_brake=True
                    ))
                    return False

            # Calculate and apply control
            control = self._calculate_control(current_speed, vehicle_transform, dt)
            self.vehicle.apply_control(control)
            return True

        except Exception as e:
            print(f"Error in control update: {e}")
            return False

    def _need_new_target(self, vehicle_transform):
        """Check if we need to update the target waypoint"""
        return (self.current_target is None or 
                self._is_target_reached(vehicle_transform.location, self.current_target.location))

    def _update_target(self):
        """Get next target waypoint, return False if route is complete"""
        if self.route_manager.has_more_waypoints():
            self.current_target = self.route_manager.get_next_waypoint()
            return True
        else:
            # Route completed, apply full brake
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True
            ))
            print("\nRoute completed - Vehicle stopped")
            return False

    def _calculate_control(self, current_speed, vehicle_transform, dt):
        """Calculate control signals based on current state"""
        # Update future waypoints
        self.future_waypoints = self._get_future_waypoints()
        
        if not self.future_waypoints:
            return carla.VehicleControl(brake=1.0)

        # Calculate weighted target point based on multiple waypoints
        target_location = self._calculate_target_point(vehicle_transform)
        
        # Speed control with braking
        speed_error = self.target_speed - current_speed
        
        if abs(speed_error) < CONFIG['SPEED_THRESHOLD']:
            # Maintain current speed
            throttle = self.throttle_controller.run_step(speed_error, dt)
            brake = 0.0
        elif speed_error > 0:
            # Need to accelerate
            throttle = self.throttle_controller.run_step(speed_error, dt)
            brake = 0.0
        else:
            # Need to slow down
            throttle = 0.0
            brake = min(abs(speed_error) * 0.1, 1.0)  # Proportional braking

        throttle = np.clip(throttle, 0.0, 1.0)

        # Steering control with multiple waypoint consideration
        direction_vector = target_location - vehicle_transform.location
        self.next_waypoint_distance = math.sqrt(direction_vector.x**2 + direction_vector.y**2)
        
        # Adjust target speed based on distance to next waypoint
        target_speed_factor = min(self.next_waypoint_distance / 5.0, 1.0)
        speed_limit = self.target_speed * target_speed_factor
        
        if current_speed > speed_limit:
            throttle = 0.0
            brake = 0.3
        
        # Calculate steering
        forward_vector = vehicle_transform.get_forward_vector()
        steering_error = math.atan2(direction_vector.y, direction_vector.x) - math.atan2(forward_vector.y, forward_vector.x)
        # Normalize steering error to [-pi, pi]
        steering_error = math.atan2(math.sin(steering_error), math.cos(steering_error))
        steering = self.steering_controller.run_step(steering_error, dt)
        steering = np.clip(steering, -1.0, 1.0)

        # Calculate and store lateral error (add after calculating direction_vector)
        lateral_error = self._calculate_lateral_error(vehicle_transform)
        self.lateral_error_history.append(lateral_error)

        return carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steering),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )

    def _get_future_waypoints(self):
        """Get next few waypoints for path planning"""
        future_points = []
        current_index = self.route_manager.current_waypoint_index - 1
        
        for i in range(CONFIG['PATH_PLANNING']['LOOKAHEAD_POINTS']):
            next_point = self.route_manager.peek_waypoint(current_index + i)
            if next_point:
                future_points.append(next_point)
            
        return future_points

    def _calculate_target_point(self, vehicle_transform):
        """Calculate weighted target point based on multiple waypoints"""
        if not self.future_waypoints:
            return self.current_target.location

        # Initialize with first waypoint to avoid division by zero
        weighted_location = carla.Location(
            x=self.future_waypoints[0].location.x,
            y=self.future_waypoints[0].location.y,
            z=self.future_waypoints[0].location.z
        )
        
        # If we have multiple waypoints, apply weighted average
        if len(self.future_waypoints) > 1:
            total_weight = 0
            weighted_location = carla.Location(x=0, y=0, z=0)
            
            for point, weight in zip(self.future_waypoints, CONFIG['PATH_PLANNING']['WEIGHTS']):
                weighted_location.x += point.location.x * weight
                weighted_location.y += point.location.y * weight
                weighted_location.z += point.location.z * weight
                total_weight += weight

            # Only normalize if we have valid weights
            if total_weight > 0:
                weighted_location.x /= total_weight
                weighted_location.y /= total_weight
                weighted_location.z /= total_weight

        return weighted_location

    def _is_target_reached(self, current_location, target_location, threshold=2.0):
        """Check if we've reached the target waypoint."""
        distance = current_location.distance(target_location)
        # Increase threshold if we're going too fast
        adjusted_threshold = threshold * (1.0 + (self.get_speed() / self.target_speed))
        return distance < adjusted_threshold

    def _calculate_lateral_error(self, vehicle_transform):
        """Calculate lateral error considering vehicle orientation and path curvature"""
        if not self.future_waypoints or len(self.future_waypoints) < 2:
            return 0.0

        # Get vehicle's coordinate system
        vehicle_location = vehicle_transform.location
        vehicle_forward = vehicle_transform.get_forward_vector()
        vehicle_right = vehicle_transform.get_right_vector()

        # Transform waypoints to vehicle's local coordinates
        wp1 = self.future_waypoints[0].location
        wp2 = self.future_waypoints[1].location

        # Calculate vectors in vehicle's coordinate system
        to_first_waypoint = carla.Vector3D(
            wp1.x - vehicle_location.x,
            wp1.y - vehicle_location.y,
            0
        )

        # Project first waypoint onto vehicle's axes
        forward_proj = (to_first_waypoint.x * vehicle_forward.x + 
                       to_first_waypoint.y * vehicle_forward.y)
        right_proj = (to_first_waypoint.x * vehicle_right.x + 
                     to_first_waypoint.y * vehicle_right.y)

        # Calculate path direction in vehicle's coordinate system
        path_vector = carla.Vector3D(
            wp2.x - wp1.x,
            wp2.y - wp1.y,
            0
        )
        path_length = math.sqrt(path_vector.x**2 + path_vector.y**2)
        
        if path_length < 0.001:
            return right_proj  # Return direct lateral offset if path segment is too short

        # Calculate path angle relative to vehicle
        path_forward_proj = (path_vector.x * vehicle_forward.x + 
                           path_vector.y * vehicle_forward.y) / path_length
        path_angle = math.acos(np.clip(path_forward_proj, -1.0, 1.0))

        # Adjust lateral error based on path angle
        # Reduce lateral error as path angle increases (during turns)
        angle_factor = math.cos(path_angle)
        
        # Calculate base lateral error
        lateral_error = right_proj
        
        # Apply angle-based correction
        corrected_error = lateral_error * (0.5 + 0.5 * abs(angle_factor))
        
        # Determine sign based on which side of the path the vehicle is on
        cross_product = (vehicle_forward.x * path_vector.y - 
                        vehicle_forward.y * path_vector.x)
        
        return corrected_error * (1 if cross_product >= 0 else -1)

    def analyze_speed_performance(self):
        """Analyze time spent within and outside speed threshold"""
        target = CONFIG['SPEED_KMH']
        threshold = CONFIG['SPEED_THRESHOLD'] + 0.5  # Modified threshold for analysis
        total_time = self.time_history[-1] - self.time_history[0]
        
        # Count time within threshold
        time_in_range = sum(
            t2 - t1 
            for s1, s2, t1, t2 in zip(
                self.speed_history[:-1], 
                self.speed_history[1:],
                self.time_history[:-1],
                self.time_history[1:]
            )
            if abs(s1 - target) <= threshold
        )
        
        time_outside = total_time - time_in_range
        
        # Create performance plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_history, self.speed_history, 'b-', label='Actual Speed')
        plt.axhline(y=target, color='g', linestyle='--', label='Target Speed')
        plt.axhline(y=target + threshold, color='r', linestyle=':', label='Threshold')
        plt.axhline(y=target - threshold, color='r', linestyle=':')
        plt.fill_between(self.time_history, 
                        [target - threshold] * len(self.time_history),
                        [target + threshold] * len(self.time_history),
                        alpha=0.2, color='g')
        
        plt.title('Vehicle Speed Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (km/h)')
        plt.legend()
        plt.grid(True)
        
        # Add performance text
        performance_text = (
            f'Time in range: {time_in_range:.1f}s ({(time_in_range/total_time)*100:.1f}%)\n'
            f'Time outside: {time_outside:.1f}s ({(time_outside/total_time)*100:.1f}%)'
        )
        plt.text(0.02, 0.98, performance_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return plt.gcf()

    def analyze_lateral_error(self):
        """Create plot of lateral error over time"""
        if not self.lateral_error_history or not self.time_history:
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(self.time_history[:len(self.lateral_error_history)], 
                self.lateral_error_history, 'b-', label='Lateral Error')
        plt.axhline(y=0, color='g', linestyle='--', label='Center Line')
        
        max_error = max(abs(min(self.lateral_error_history)), 
                       abs(max(self.lateral_error_history)))
        threshold = 1.0  # 1 meter threshold
        plt.axhline(y=threshold, color='r', linestyle=':', label='Threshold')
        plt.axhline(y=-threshold, color='r', linestyle=':')
        
        plt.fill_between(self.time_history[:len(self.lateral_error_history)], 
                        [-threshold] * len(self.lateral_error_history),
                        [threshold] * len(self.lateral_error_history),
                        alpha=0.2, color='g')
        
        plt.title('Lateral Error Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Lateral Error (m)')
        plt.legend()
        plt.grid(True)
        
        # Calculate statistics
        time_in_range = sum(
            t2 - t1 
            for e1, e2, t1, t2 in zip(
                self.lateral_error_history[:-1],
                self.lateral_error_history[1:],
                self.time_history[:-1],
                self.time_history[1:]
            )
            if abs(e1) <= threshold
        )
        
        total_time = self.time_history[-1] - self.time_history[0]
        time_outside = total_time - time_in_range
        
        performance_text = (
            f'Time in range: {time_in_range:.1f}s ({(time_in_range/total_time)*100:.1f}%)\n'
            f'Time outside: {time_outside:.1f}s ({(time_outside/total_time)*100:.1f}%)\n'
            f'Max error: {max_error:.2f}m'
        )
        
        plt.text(0.02, 0.98, performance_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return plt.gcf()

    def destroy(self):
        for actor in self.actor_list:
            actor.destroy()

def run_scenario(client, world, route_file):
    """Run a single scenario with given route file"""
    actor_list = []
    route_manager = None
    
    try:
        # Get spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_index = min(CONFIG['SPAWN_INDEX'], len(spawn_points) - 1)
        start_point = spawn_points[spawn_index]

        # Initialize route and vehicle
        route_manager = RouteManager(os.path.join(CONFIG['WAYPOINTS_BASE_PATH'], route_file))
        route_manager.prepend_spawn_point(start_point)
        route_manager.initialize_route(world)

        # Position spectator camera at fixed position
        spectator = world.get_spectator()
        camera_location = carla.Location(x=0, y=200, z=100)
        camera_rotation = carla.Rotation(pitch=-45.0)
        spectator.set_transform(carla.Transform(camera_location, camera_rotation))
        print("\nCamera positioned at fixed viewpoint")

        # Spawn vehicle
        car = SpawnCar(world, world.get_blueprint_library(), start_point, route_manager)
        actor_list.append(car)

        # Control loop
        running = True
        while running:
            running = car.update_control(CONFIG['CONTROL_RATE'])
            if not running:
                break
            time.sleep(CONFIG['SLEEP_TIME'])

        # After completion, analyze and save performance plots
        if not car.collision_detected:
            # Speed performance
            plt.figure(figsize=(10, 6))
            performance_plot = car.analyze_speed_performance()
            plot_filename = f"speed_performance_{os.path.splitext(route_file)[0]}.png"
            performance_plot.savefig(os.path.join(CONFIG['WAYPOINTS_BASE_PATH'], plot_filename))
            plt.close()
            
            # Lateral error performance
            lateral_plot = car.analyze_lateral_error()
            if lateral_plot:
                plot_filename = f"lateral_error_{os.path.splitext(route_file)[0]}.png"
                lateral_plot.savefig(os.path.join(CONFIG['WAYPOINTS_BASE_PATH'], plot_filename))
                plt.close()

        return True

    except Exception as e:
        print(f"\nError in scenario {route_file}: {e}")
        return False
    finally:
        print('\nCleaning up scenario...')
        for actor in actor_list:
            if actor and hasattr(actor, 'destroy'):
                actor.destroy()

def main():
    try:
        # Setup CARLA connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Run each scenario
        for scenario_file in CONFIG['SCENARIOS']:
            print(f"\n\nStarting scenario with route: {scenario_file}")
            print("----------------------------------------")
            
            # Reload the world to start fresh
            world = client.load_world(CONFIG['TOWN'])
            world.tick()
            time.sleep(1.0)  # Wait for world to stabilize
            
            # Run the scenario
            success = run_scenario(client, world, scenario_file)
            
            if success:
                print(f"Scenario {scenario_file} completed successfully")
            else:
                print(f"Scenario {scenario_file} failed")
            
            time.sleep(1.0)  # Pause between scenarios

    except KeyboardInterrupt:
        print("\nStopping simulation (Ctrl+C pressed)")
    except Exception as e:
        print(f"\nError in main: {e}")
    finally:
        print('Done.')

if __name__ == '__main__':
    main()
