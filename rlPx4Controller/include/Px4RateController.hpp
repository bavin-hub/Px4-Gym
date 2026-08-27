#ifndef __PX4_RATE_CONTROLLER
#define __PX4_RATE_CONTROLLER
#include <Eigen/Eigen>
// #include <pybind11/eigen.h>
#include "MyMath.hpp"
/*
 * Note: order of axis are match tf2::LinearMath (bullet).
 * YPR rotation convention -> YAW first, Pitch second, Roll third
 * Compatibility checked by unittests.
 */

Eigen::Quaterniond quaternion_from_rpy(const Eigen::Vector3d &rpy)
{
	// YPR - ZYX
	return Eigen::Quaterniond(
		Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()));
}

/**
 * @brief Convert euler angles to quaternion.
 *
 * @return quaternion, same as @a tf::quaternionFromRPY() but in Eigen format.
 */
inline Eigen::Quaterniond quaternion_from_rpy(const double roll, const double pitch, const double yaw)
{
	return quaternion_from_rpy(Eigen::Vector3d(roll, pitch, yaw));
}
static const auto NED_ENU_Q = quaternion_from_rpy(M_PI, 0.0, M_PI_2);
static const Eigen::Affine3d NED_ENU_AFFINE(NED_ENU_Q);

class Px4RateController
{
private:
	/**
	 * @brief Static vector needed for rotating between ENU and NED frames
	 * +PI rotation around X (North) axis follwed by +PI/2 rotation about Z (Down)
	 * gives the ENU frame.  Similarly, a +PI rotation about X (East) followed by
	 * a +PI/2 roation about Z (Up) gives the NED frame.
	 */

	Eigen::Vector3d _gain_p; ///< rate control proportional gain for all axes x, y, z
	Eigen::Vector3d _gain_i; ///< rate control integral gain
	Eigen::Vector3d _gain_d; ///< rate control derivative gain

	Eigen::Vector3d _lim_int; ///< integrator term maximum absolute value

	// States
	Eigen::Vector3d _rate_int; ///< integral term of the rate controller

	/* data */

	Eigen::Vector3d _rate_target;
	Eigen::Vector3d _rate_now;

public:
	Px4RateController(/* args */);
	~Px4RateController();
	Eigen::Vector3d update(const Eigen::Vector3d &rate_sp, const Eigen::Vector3d &rate, const Eigen::Vector3d &angular_accel,
						   const float dt)
	{

		return update(rate, rate_sp, angular_accel, dt,false);
	}
	void set_q_world(const Eigen::Vector4d &q)
	{
		q_world = Eigen::Quaterniond(q(0),q(1),q(2),q(3));
	}

	Eigen::Vector3d update(const Eigen::Vector3d &rate, const Eigen::Vector3d &rate_sp, const Eigen::Vector3d &angular_accel,
						   const float dt, const bool landed);
	void updateIntegral(Eigen::Vector3d &rate_error, const float dt);
	void set_pid_params(Eigen::Vector3d gain_p, Eigen::Vector3d gain_i, Eigen::Vector3d gain_d);

	
	Eigen::Vector3d get_rate_target();
	Eigen::Vector3d get_rate_now();
	Eigen::Vector3d transform_world2body(Eigen::Quaterniond q, Eigen::Vector3d rate_world);
	Eigen::Quaterniond q_world;
};

Px4RateController::Px4RateController()
{
	set_pid_params(Eigen::Vector3d(0.5,0.5,0.2),
				   Eigen::Vector3d(0.08,0.08,0.05),
				   Eigen::Vector3d(0.001,0.001,0.0));
}
void Px4RateController::set_pid_params(Eigen::Vector3d gain_p, Eigen::Vector3d gain_i, Eigen::Vector3d gain_d)
{
	// x ,y ,z
	// roll , pitch , yaw
	_lim_int << 0.3, 0.3, 0.3;
	_gain_p = gain_p;
	_gain_i = gain_i;
	_gain_d = gain_d;
	_rate_int << 0.0, 0.0, 0.0;
}

Px4RateController::~Px4RateController()
{
}

Eigen::Vector3d Px4RateController::transform_world2body(Eigen::Quaterniond q, Eigen::Vector3d rate_world)
{
	Eigen::Matrix3d rotationMatrix = q.toRotationMatrix().transpose();
	// // XYZ  z-y-x的顺序
	// double roll = atan2(-rotationMatrix(1, 2), rotationMatrix(2, 2));
	// double pitch = asin(rotationMatrix(0, 2));
	// double yaw = atan2(-rotationMatrix(0, 1), rotationMatrix(0, 0));
	// std::cout << "roll pitch yaw " <<roll<<" "<<pitch<<" "<<yaw<<std::endl;

	// // ENU frame
	// Eigen::Matrix3d Rotation_matrix;
	// Rotation_matrix << 1, 0, -sin(pitch),
	// 	0, cos(roll), cos(pitch) * sin(roll),
	// 	0, -sin(roll), cos(pitch) * cos(roll);
	Eigen::Vector3d body_rate;
	body_rate = rotationMatrix * rate_world;
	return body_rate;
}

Eigen::Vector3d Px4RateController::update(const Eigen::Vector3d &rate, const Eigen::Vector3d &rate_sp, const Eigen::Vector3d &angular_accel,
										  const float dt, const bool landed)
{
	Eigen::Vector3d body_rate = transform_world2body(q_world, rate);

	// static const Eigen::PermutationMatrix<3> NED_ENU_REFLECTION_XY(Eigen::Vector3i(1,0,2));
	// static const Eigen::DiagonalMatrix<double,3> NED_ENU_REFLECTION_Z(1,1,-1);

	// Eigen::Vector3d rate_enu = NED_ENU_REFLECTION_XY * (NED_ENU_REFLECTION_Z * body_rate);
	// Eigen::Vector3d rate_sp_enu = NED_ENU_REFLECTION_XY * (NED_ENU_REFLECTION_Z * rate_sp);

	// angular rates error
	Eigen::Vector3d rate_error = rate_sp - body_rate;

	// Eigen::Vector3d body_rate = transform_world2body(q_world,rate);
	// rate_error = transform_world2body(q_world,rate_error);

	// std::cout << "rate_sp" <<rate_sp(0)<<" "<<rate_sp(1)<<" "<<rate_sp(2)<<std::endl;
	// std::cout << "body_rate" <<body_rate(0)<<" "<<body_rate(1)<<" "<<body_rate(2)<<std::endl;

	// std::cout << "rate_error" <<rate_error(0)<<" "<<rate_error(1)<<" "<<rate_error(2)<<std::endl;

	// PID control with feed forward
	const Eigen::Vector3d torque = _gain_p.cwiseProduct(rate_error) + _rate_int - _gain_d.cwiseProduct(angular_accel);

	// update integral only if we are not landed
	if (!landed)
	{
		updateIntegral(rate_error, dt);
	}


	return torque;
}
// rate_sp1.12452 0.824669 -11.6932

void Px4RateController::updateIntegral(Eigen::Vector3d &rate_error, const float dt)
{
	for (int i = 0; i < 3; i++)
	{
		// prevent further positive control saturation
		// if (_control_allocator_saturation_positive(i)) {
		// 	rate_error(i) = math::min(rate_error(i), 0.f);
		// }

		// // prevent further negative control saturation
		// if (_control_allocator_saturation_negative(i)) {
		// 	rate_error(i) = math::max(rate_error(i), 0.f);
		// }

		// I term factor: reduce the I gain with increasing rate error.
		// This counteracts a non-linear effect where the integral builds up quickly upon a large setpoint
		// change (noticeable in a bounce-back effect after a flip).
		// The formula leads to a gradual decrease w/o steps, while only affecting the cases where it should:
		// with the parameter set to 400 degrees, up to 100 deg rate error, i_factor is almost 1 (having no effect),
		// and up to 200 deg error leads to <25% reduction of I.
		float i_factor = rate_error(i) / MyMath::radians(400.f);
		i_factor = MyMath::max(0.0f, 1.f - i_factor * i_factor);

		// Perform the integration using a first order method
		double rate_i = _rate_int(i) + i_factor * _gain_i(i) * rate_error(i) * dt;

		// do not propagate the result if out of range or invalid
		_rate_int(i) = MyMath::constrain(rate_i, -_lim_int(i), _lim_int(i));
	}
}
#endif