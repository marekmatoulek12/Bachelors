#!/usr/bin/env bash
# Source me:  source env.sh
# Puts the SDK's Python modules and shared libraries on the right paths and
# sources a ROS 2 environment.
sdk_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${AMENT_PREFIX_PATH:-}" ]]; then
  for setup in /opt/ros/jazzy/setup.bash /opt/ros/humble/setup.bash /opt/ros/*/setup.bash; do
    [[ -f "$setup" ]] && { source "$setup"; break; }
  done
fi
export PYTHONPATH="$sdk_dir/exercises/1_path_tracking:$sdk_dir/exercises/2_state_estimation:$sdk_dir/exercises/3_low_level_control:${PYTHONPATH:-}"
export PYTHONPATH="${sdk_dir}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${sdk_dir}/lib:${LD_LIBRARY_PATH:-}"
[[ ! -f "${sdk_dir}/config/student.env" ]] || source "${sdk_dir}/config/student.env"
echo "CogniCar SDK environment ready (PYTHONPATH, LD_LIBRARY_PATH, ROS ${ROS_DISTRO:-unsourced})."
