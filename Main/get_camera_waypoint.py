import carla
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import math

# Configuration constants
SAVE_DIR = "/home/teixeia/projects/homeworks/CPS/CARLA"
FIXED_Z = 0.5
MIN_DISTANCE = 5.0
COLLECTION_INTERVAL = 0.1
DECIMAL_PRECISION = 1

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

    def save_to_xml(self, filename="waypoints.xml"):
        processed = self.process_waypoints()
        if not processed:
            print("\nNo waypoints to save!")
            return

        filepath = os.path.join(self.save_directory, filename)
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
        world = client.get_world()
        collector = WaypointCollector()
        
        print("Recording camera positions...")
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