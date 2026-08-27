#ifndef __PARALLEL_POS_CONTROL_HPP_
#define __PARALLEL_POS_CONTROL_HPP_
#include "SimplePositionController.hpp"
#include "Px4AttitudeController.hpp"
#include "Px4RateController.hpp"
#include "Px4Mixer.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "droneStatus.hpp"

class pyParallelPositionControl
{
private:
    std::vector<DroneStatus_t> _drons_stats;

    std::vector<SimplePositionController> _pos_control;
    std::vector<Px4AttitudeController> _atti_control;
    std::vector<Px4RateController> _rate_control;
    std::vector<Px4Mixer> _mixers;
    int _envs_num = 0;
    Eigen::MatrixXd _commands;

public:

    pyParallelPositionControl(int envs_num);
    void set_status(Eigen::MatrixXd pos_matrix, Eigen::MatrixXd q_matrix, Eigen::MatrixXd vel_matrix, Eigen::MatrixXd ang_vel_matrix, double dt);
    Eigen::MatrixXd update(const Eigen::MatrixXd &actions);
    ~pyParallelPositionControl()
    {

    };
};

pyParallelPositionControl::pyParallelPositionControl(int envs_num)
{
    _envs_num = envs_num;
    _commands.resize(_envs_num, 4);

    for (int i = 0; i < _envs_num; i++)
    {
        _drons_stats.push_back(DroneStatus_t());
        _pos_control.push_back(SimplePositionController());
        _pos_control.at(i).set_control_mode(control_mode::CTRL_ALL);//There cannot be only position gain P
        _atti_control.push_back(Px4AttitudeController());
        _rate_control.push_back(Px4RateController());
        _mixers.push_back(Px4Mixer());
    }
}


void pyParallelPositionControl::set_status(Eigen::MatrixXd pos_matrix, Eigen::MatrixXd q_matrix, Eigen::MatrixXd vel_matrix, Eigen::MatrixXd ang_vel_matrix, double dt)
{
    assert(pos_matrix.rows() == _envs_num && pos_matrix.cols() == 3);
    assert(q_matrix.rows() == _envs_num && q_matrix.cols() == 4);
    assert(vel_matrix.rows() == _envs_num && vel_matrix.cols() == 3);
    assert(ang_vel_matrix.rows() == _envs_num && ang_vel_matrix.cols() == 3);

    for (int i = 0; i < _envs_num; i++)
    {
        _drons_stats.at(i).position = pos_matrix.block(i, 0, i+1, 3).transpose();
        _drons_stats.at(i).velocity = vel_matrix.block(i, 0, i+1, 3).transpose();
        _drons_stats.at(i).quaternion = q_matrix.block(i, 0, i+1, 4).transpose();
        _drons_stats.at(i).angle_velocity = ang_vel_matrix.block(i, 0, i+1, 3).transpose();
        _drons_stats.at(i).dt = dt;

        _pos_control.at(i).set_status(pos_matrix.block(i, 0, i+1, 3).transpose(), vel_matrix.block(i, 0, i+1, 3).transpose(), ang_vel_matrix.block(i, 0, i+1, 3).transpose(), q_matrix.block(i, 0, i+1, 4).transpose(), dt);
        _rate_control.at(i).set_q_world(q_matrix.block(i, 0, i+1, 4).transpose());
    }
}
Eigen::MatrixXd pyParallelPositionControl::update(const Eigen::MatrixXd &actions)
{
    assert(actions.rows() == _envs_num && actions.cols() == 4);

    for (int i = 0; i < _envs_num; i++)
    {
        Eigen::Vector3d action_pos = actions.block(i, 0, i + 1, 3).transpose();
        double action_yaw = actions(i, 3);

        Eigen::VectorXd atti_thrust_sp = _pos_control.at(i).update(action_pos, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), action_yaw);

        Eigen::Vector3d rate_sp = _atti_control.at(i).update(atti_thrust_sp.head<4>(), _drons_stats.at(i).quaternion);
        Eigen::Vector3d torque_sp = _rate_control.at(i).update(rate_sp, _drons_stats.at(i).angle_velocity, Eigen::Vector3d(0, 0, 0), _drons_stats.at(i).dt);

        Eigen::Vector4d torque_thrust_sp = Eigen::Vector4d(torque_sp(0), torque_sp(1), torque_sp(2), atti_thrust_sp(4));

        Eigen::Vector4d cmd = _mixers.at(i).update(torque_thrust_sp);

        _commands.block(i,0,i+1,4) << cmd(0) , cmd(1), cmd(2), cmd(3); 
    }
    return _commands;
}
#endif