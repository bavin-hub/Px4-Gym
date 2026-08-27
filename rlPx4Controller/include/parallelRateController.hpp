#ifndef __PARALLEL_RATE_CONTROL_HPP_
#define __PARALLEL_RATE_CONTROL_HPP_
#include "Px4RateController.hpp"
#include "Px4Mixer.hpp"

class pyParallelRateControl
{
private:
    std::vector<Px4RateController> _rate_control;
    std::vector<Px4Mixer> _mixers;
    int _envs_num = 0;
    Eigen::MatrixXd _commands;

public:
    pyParallelRateControl(int envs_num);
    void set_q_world(Eigen::MatrixXd q_matrix);
    Eigen::MatrixXd update(const Eigen::MatrixXd &actions, const Eigen::MatrixXd &observations, double dt);
    ~pyParallelRateControl();
};

pyParallelRateControl::pyParallelRateControl(int envs_num)
{
    _envs_num = envs_num;
    _commands.resize(_envs_num, 4);

    for (int i = 0; i < _envs_num; i++)
    {
        _rate_control.push_back(Px4RateController());
        _mixers.push_back(Px4Mixer());
    }
}
void pyParallelRateControl::set_q_world(Eigen::MatrixXd q_matrix)
{
    assert(q_matrix.rows() == _envs_num && q_matrix.cols() == 4);
    for (int i = 0; i < _envs_num; i++)
        _rate_control.at(i).set_q_world(q_matrix.block(i, 0, i, 4).transpose());
}
Eigen::MatrixXd pyParallelRateControl::update(const Eigen::MatrixXd &actions, const Eigen::MatrixXd &observations, double dt)
{
    assert(actions.rows() == _envs_num && actions.cols() == 4);
    assert(observations.rows() == _envs_num && observations.cols() == 3);

    for (int i = 0; i < _envs_num; i++)
    {
        Eigen::Vector3d action_rate = actions.block(i, 0, i + 1, 3).transpose();
        Eigen::Vector3d rate = observations.block(i, 0, i + 1, 3).transpose();
        Eigen::Vector3d torque_sp = _rate_control.at(i).update(action_rate, rate, Eigen::Vector3d(0, 0, 0), dt);
        Eigen::Vector4d torque_thrust_sp = Eigen::Vector4d(torque_sp(0), torque_sp(1), torque_sp(2), actions(i, 3));

        Eigen::Vector4d cmd = _mixers.at(i).update(torque_thrust_sp);

        _commands.block(i, 0, i + 1, 4) << cmd(0), cmd(1), cmd(2), cmd(3);
        // _commands.block(i,0,i,4) = _mixers.at(i).update(Eigen::Vector4d(torque_sp(0),torque_sp(1),torque_sp(2),actions(i,3)));
        //     // std::cout<< commands  << std::endl;
    }

    // py::array_t<double> result({_commands.rows(),_commands.cols()}, _commands.data());
    return _commands;
}
pyParallelRateControl::~pyParallelRateControl()
{
}

#endif