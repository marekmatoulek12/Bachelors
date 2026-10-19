import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/marekm/CVUT/Bakalarka/ros2_ws/install/car_tests_pkg'
