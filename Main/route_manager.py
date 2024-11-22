import carla
import numpy as np
import xml.etree.ElementTree as ET

class RouteManager:
    def __init__(self, xml_file=None):
        self.world = None
        self.waypoints = []
        self.current_waypoint_index = 0
        self.debug_objects = []  # Add this line
        if xml_file:
            self.load_from_xml(xml_file)

    def prepend_spawn_point(self, spawn_transform):
        """Add spawn point as the first waypoint in the route"""
        if not isinstance(spawn_transform, carla.Transform):
            raise TypeError("spawn_transform must be a carla.Transform")
        
        # Insert spawn point at the beginning of waypoints list
        self.waypoints.insert(0, spawn_transform)
        print(f"\nAdded spawn point as first waypoint: x={spawn_transform.location.x:.1f}, y={spawn_transform.location.y:.1f}, z={spawn_transform.location.z:.1f}")

    def _draw_debug_visualizations(self):
        """Draw all debug visualizations for waypoints"""
        if not self.world:
            return

        # Clean previous visualizations
        self.clear_debug_objects()
        
        debug = self.world.debug
        for i in range(len(self.waypoints)):
            location = self.waypoints[i].location
            self.debug_objects.append(
                debug.draw_point(
                    location,
                    size=0.2,
                    color=carla.Color(r=0, g=255, b=0),
                    life_time=0.0
                )
            )
            
            if i < len(self.waypoints) - 1:
                next_location = self.waypoints[i + 1].location
                self.debug_objects.append(
                    debug.draw_line(
                        location,
                        next_location,
                        thickness=0.1,
                        color=carla.Color(r=0, g=255, b=0),
                        life_time=0.0
                    )
                )

    def clear_debug_objects(self):
        """Clean up debug visualization objects"""
        for obj in self.debug_objects:
            try:
                if obj:
                    obj.destroy()
            except:
                pass
        self.debug_objects = []

    def destroy(self):
        """Clean up all resources"""
        self.clear_debug_objects()
        self.waypoints = []
        self.current_waypoint_index = 0

    def load_from_xml(self, xml_file):
        """Load waypoints from XML file."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        print("\nLoading waypoints from XML file:")
        print("-------------------------------------------")
        for waypoint in root.findall('waypoint'):
            x = float(waypoint.get('x'))
            y = float(waypoint.get('y'))
            z = float(waypoint.get('z'))
            self.add_custom_waypoint(x, y, z)
            # print(f"Loaded waypoint: x={x}, y={y}, z={z}") # If you want to see all waypoints

    def add_custom_waypoint(self, x, y, z, rotation=None):
        """Add a waypoint at specified coordinates."""
        location = carla.Location(x=x, y=y, z=z)
        if rotation is None:
            rotation = carla.Rotation()
        transform = carla.Transform(location, rotation)
        self.waypoints.append(transform)

    def initialize_route(self, world):
        """Initialize route with world reference."""
        self.world = world
        self.current_waypoint_index = 0
        self._draw_debug_visualizations()

    def add_waypoint(self, waypoint):
        """Add a single waypoint to the route."""
        if isinstance(waypoint, (int, np.integer)):  # Accept any integer type
            self.waypoint_indices.append(waypoint)
        elif isinstance(waypoint, carla.Transform):
            self.waypoints.append(waypoint)
        else:
            raise TypeError("Waypoint must be either an integer index or carla.Transform")

    def add_waypoints(self, waypoints):
        """Add multiple waypoints to the route."""
        for waypoint in waypoints:
            self.add_waypoint(waypoint)

    def clear_route(self):
        """Clear all waypoints from the route."""
        self.waypoints = []
        self.current_waypoint_index = 0

    def get_next_waypoint(self):
        """Get the next waypoint in the route."""
        if not self.waypoints:
            return None
        
        if self.current_waypoint_index < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_index]
            self.current_waypoint_index += 1
            return waypoint
        return None

    def has_more_waypoints(self):
        """Check if there are more waypoints in the route."""
        return self.current_waypoint_index < len(self.waypoints)

    def reset_route(self):
        """Reset the route to start from the beginning."""
        self.current_waypoint_index = 0

    def get_route_length(self):
        """Get the total number of waypoints in the route."""
        return len(self.waypoints)

    def get_remaining_waypoints(self):
        """Get the number of remaining waypoints."""
        return len(self.waypoints) - self.current_waypoint_index

    def peek_waypoint(self, index):
        """Look at a future waypoint without advancing the counter"""
        if 0 <= index < len(self.waypoints):
            return self.waypoints[index]
        return None