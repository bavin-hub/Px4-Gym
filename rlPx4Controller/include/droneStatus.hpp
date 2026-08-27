#ifndef __DRONE_STATUS_HPP_
#define __DRONE_STATUS_HPP_
#include <Eigen/Eigen>

struct DroneStatus_t
{
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector4d quaternion; // w x y z
    Eigen::Vector3d angle_velocity;
    double dt;

    DroneStatus_t() : position(Eigen::Vector3d::Zero()), velocity(Eigen::Vector3d::Zero())
    ,quaternion(Eigen::Vector4d::Zero()), angle_velocity(Eigen::Vector3d::Zero()) {
    }
};
#endif