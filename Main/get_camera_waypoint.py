import carla
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import math
import glob

# Configuration constants
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_Z = 0.5
MIN_DISTANCE = 5.0
COLLECTION_INTERVAL = 0.1
DECIMAL_PRECISION = 1
SPAWN_INDEX = 73  # Match spawn index from spawn_car.py
TOWN_INDEX = 'Town02'

class WaypointCollector:
    def __init__(self):
        self.raw_waypoints = []
        self.save_directory = SAVE_DIR
        self.FIXED_Z = FIXED_Z
        self.MIN_DISTANCE = MIN_DISTANCE
        self.points_collected = 0
        
    def add_waypoint(self, x, y, z):
        """Add waypoint coordinates with fixed z height"""
        self.raw_waypoints.append((
            round(x, DECIMAL_PRECISION),
            round(y, DECIMAL_PRECISION),
            self.FIXED_Z
        ))
        self.points_collected += 1
        
    def process_waypoints(self):
        """Process waypoints to maintain minimum distance while preserving path shape"""
        if not self.raw_waypoints:
            return []
        
        filtered = [self.raw_waypoints[0]]
        last_kept = self.raw_waypoints[0]
        
        for point in self.raw_waypoints[1:]:
            distance = math.sqrt(
                (point[0] - last_kept[0])**2 +
                (point[1] - last_kept[1])**2
            )
            
            if distance >= self.MIN_DISTANCE:
                filtered.append(point)
                last_kept = point
                
        return filtered

    def get_progress_stats(self):
        """Get collection progress statistics"""
        return {
            'total_collected': self.points_collected,
            'processed_count': len(self.process_waypoints()),
            'raw_count': len(self.raw_waypoints)
        }

    def save_to_xml(self):
        processed = self.process_waypoints()
        if not processed:
            print("\nNo waypoints to save!")
            return

        # Find the next available filename
        base_filename = "waypoints"
        extension = ".xml"
        existing_files = glob.glob(os.path.join(self.save_directory, base_filename + "*" + extension))
        max_num = 0
        for filename in existing_files:
            basename = os.path.basename(filename)
            num_str = basename.replace(base_filename, "").replace(extension, "")
            if num_str.isdigit():
                num = int(num_str)
                if num > max_num:
                    max_num = num
        next_num = max_num + 1
        if (next_num == 1):
            filepath = os.path.join(self.save_directory, base_filename + extension)
        else:
            filepath = os.path.join(self.save_directory, f"{base_filename}{next_num}{extension}")

        root = ET.Element("waypoints")
        
        for i, (x, y, z) in enumerate(processed):
            point = ET.SubElement(root, "waypoint")
            point.set("id", str(i))
            point.set("x", str(x))
            point.set("y", str(y))
            point.set("z", str(z))
        
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        
        try:
            with open(filepath, "w") as f:
                f.write(xml_str)
            stats = self.get_progress_stats()
            print(f"\nCollection summary:")
            print(f"Total points collected: {stats['total_collected']}")
            print(f"Points after processing: {stats['processed_count']}")
            print(f"Saved to: {filepath}")
        except Exception as e:
            print(f"\nError saving waypoints: {e}")

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Load Town02
        world = client.load_world(TOWN_INDEX)
        world.tick()
        time.sleep(1.0)  # Wait for world to stabilize
        
        # Get the same spawn point as spawn_car.py
        spawn_points = world.get_map().get_spawn_points()
        spawn_index = min(SPAWN_INDEX, len(spawn_points) - 1)
        start_point = spawn_points[spawn_index]
        
        # Set spectator to start point
        spectator = world.get_spectator()
        spectator.set_transform(start_point)
        
        collector = WaypointCollector()
        
        print("\nPosition the camera as desired...")
        print("Press Enter when ready to start recording camera positions...")
        input()
        
        print("\nRecording camera positions...")
        print(f"- Fixed Z height: {FIXED_Z}m")
        print(f"- Minimum distance between points: {MIN_DISTANCE}m")
        print("Press Ctrl+C to stop and save...")
        
        while True:
            spectator = world.get_spectator()
            transform = spectator.get_transform()
            
            collector.add_waypoint(
                transform.location.x,
                transform.location.y,
                transform.location.z
            )
            
            stats = collector.get_progress_stats()
            print(f"\rPoints collected: {stats['total_collected']} (will save {stats['processed_count']} after processing)", end="")
            
            time.sleep(COLLECTION_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nStopping collection...")
        collector.save_to_xml()
        print("Done!")
    except Exception as e:
        print(f"\nError: {e}")
        
if __name__ == "__main__":
    main()