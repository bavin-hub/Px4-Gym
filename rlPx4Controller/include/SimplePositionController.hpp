#ifndef __SIMPLE_POSITION_CONTROLLER
#define __SIMPLE_POSITION_CONTROLLER
#include <Eigen/Eigen>
#include "HoverThrustEkf.hpp"
#include "Derivative.hpp"

enum control_mode
{
    CTRL_ALL,
    CTRL_POS_ONLY,
    CTRL_VEL_ONLY
};

double fromQuaternion2yaw(Eigen::Quaterniond q)
{
    double yaw = atan2(2 * (q.x() * q.y() + q.w() * q.z()), q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z());
    return yaw;
}
class SimplePositionController
{
private:
    HoverThrustEkf *hoverThrustEkf;
    Derivate velDerivateZ_;

    Eigen::Vector3d _pos_world;
    Eigen::Vector3d _vel_world;
    Eigen::Quaterniond _q_world;
    Eigen::Vector3d _angular_vel_world;

    double _hover_thrust{0.5}; // 不能是0

    double _thrust_sp{0.0};
    Eigen::Quaterniond q_sp;

    Eigen::Vector3d _Kp, _Kv;

    control_mode _mode = control_mode::CTRL_ALL;


public:
    void set_pid_params(Eigen::Vector3d pos_gains, Eigen::Vector3d vel_gains)
    {
        _Kp = pos_gains;
        _Kv = vel_gains;
    };
    double get_hover_thrust()
    {
        return _hover_thrust;
    };
    void set_status(Eigen::Vector3d pos, Eigen::Vector3d vel, Eigen::Vector3d angular_velocity, Eigen::Vector4d q, double dt)
    {
        _vel_world = vel;
        _pos_world = pos;
        _q_world.w() = q(0);
        _q_world.x() = q(1);
        _q_world.y() = q(2);
        _q_world.z() = q(3);
        _angular_vel_world = angular_velocity;

        // _hover_thrust = hover_thrust_estimator(0.005, _vel_world(2), thrust_z(2));
        Eigen::Vector3d thrust_z = _q_world.toRotationMatrix() * Eigen::Vector3d(0, 0, _thrust_sp);

        hoverThrustEkf->predict(dt); // dt
        double acc_z = velDerivateZ_.update(_vel_world(2), dt);
        hoverThrustEkf->fuseAccZ(acc_z, thrust_z(2));
        // _hover_thrust =  hoverThrustEkf->getHoverThrust();
        _hover_thrust = 0.1533; //0.29233965277671814;

    };
    Eigen::VectorXd update(const Eigen::Vector3d &pos_sp, const Eigen::Vector3d &vel_sp, const Eigen::Vector3d &acc_sp, const double yaw_sp);
    void set_control_mode(control_mode mode);
    SimplePositionController(/* args */);
    ~SimplePositionController();
};

SimplePositionController::SimplePositionController(/* args */)
{
    _Kp << 1.5, 1.5, 1.5;
    _Kv << 1.5, 1.5, 1.5;
    hoverThrustEkf = new HoverThrustEkf(0.4f, 0.1f, 0.0036f);
}

SimplePositionController::~SimplePositionController()
{
}

void SimplePositionController::set_control_mode(control_mode mode)
{
    _mode = mode;
}
Eigen::VectorXd SimplePositionController::update(const Eigen::Vector3d &pos_sp, const Eigen::Vector3d &vel_sp, const Eigen::Vector3d &acc_sp, const double yaw_sp)
{
    // std::cout << "pos_sp " <<pos_sp<<std::endl;
    // std::cout << "vel_sp " <<vel_sp<<std::endl;
    // std::cout << "acc_sp " <<acc_sp<<std::endl;
    // std::cout << "yaw_sp " <<yaw_sp<<std::endl;

    // compute disired acceleration
    Eigen::Vector3d des_acc(0.0, 0.0, 0.0);
    Eigen::Vector3d Kp, Kv;


    if(_mode == CTRL_POS_ONLY)
    {
        des_acc = acc_sp + _Kp.asDiagonal() * (pos_sp - _pos_world);
    }
    else if(_mode == CTRL_VEL_ONLY)
    {
        des_acc = acc_sp + _Kv.asDiagonal() * (vel_sp - _vel_world);
    // des_acc += Eigen::Vector3d(0,0,0);
    }
    else
    {
        des_acc = acc_sp + _Kv.asDiagonal() * (vel_sp - _vel_world) + _Kp.asDiagonal() * (pos_sp - _pos_world);
    }

	for (int i = 0; i < 3; i++) {
		des_acc(i) = MyMath::constrain(des_acc(i), double(-4.f),double(4.f));
	}
    // des_acc += Eigen::Vector3d(0, 0, 0);
    // std::cout << "des_acc " <<des_acc<<std::endl;


    _thrust_sp = des_acc(2) * (_hover_thrust / CONSTANTS_ONE_G) + _hover_thrust;

    double roll, pitch, yaw, yaw_imu;
    double yaw_odom = fromQuaternion2yaw(_q_world);
    double sin = std::sin(yaw_odom);
    double cos = std::cos(yaw_odom);
    roll = (des_acc(0) * sin - des_acc(1) * cos) / CONSTANTS_ONE_G;
    pitch = (des_acc(0) * cos + des_acc(1) * sin) / CONSTANTS_ONE_G;

    q_sp = Eigen::AngleAxisd(yaw_sp, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    
    Eigen::VectorXd atti_thrust_sp(5);
    atti_thrust_sp(0) = q_sp.w();
    atti_thrust_sp(1) = q_sp.x();
    atti_thrust_sp(2) = q_sp.y();
    atti_thrust_sp(3) = q_sp.z();
    atti_thrust_sp(4) = _thrust_sp;

    return atti_thrust_sp;
}
#endif