# CARLA PID PROJECT
Designed a PID controller for autonomous driving in CARLA

All the code is set to Town02. You can change this if you want at the start of the code (TOWN or TOWN_INDEX).

To use the code, you need the waypoints and supporting code in the same folder as the spawn_car.py code. Have CARLA open when running the code.

To generate more waypoints than the examples, open CARLA and run the get_camera_waypoints.py code. This code will track your camera position and create an equally spaced (5.0 meters by default) XML file.

In the simulation, you will be able to visualize the car and the planned route that he is going to follow.

route_manager.py and pid_controller.py are supporting code for spawn_car.py. The first add the spawn point of the car to the route and display the route in CARLA. The second is the PID controller used multiple times.

You can find the results in Plots and the code in Main.
