import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cogni/MarekM_Bachelors/ros2_ws/install/car_tests_pkg'
