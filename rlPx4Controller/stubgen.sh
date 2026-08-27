pkg_root_path=$(pip list -v|grep rlPx4Controller|awk '{print $3}')
cd /
python -m pybind11_stubgen rlPx4Controller.pyControl -o ${pkg_root_path}
python -m pybind11_stubgen rlPx4Controller.pyParallelControl -o ${pkg_root_path}
python -m pybind11_stubgen rlPx4Controller.traj_tools.polyTrajGen -o ${pkg_root_path}
cd -