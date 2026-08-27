#ifndef __PX4_ATTITUDE_CONTROLLER
#define __PX4_ATTITUDE_CONTROLLER
#include <Eigen/Eigen>
#include <pybind11/eigen.h>
#include "MyMath.hpp"
#include <iostream>
#include <cmath>


/**
 * Quaternion from two vectors
 * Generates shortest rotation from source to destination vector
 *
 * @param dst destination vector (no need to normalize)
 * @param src source vector (no need to normalize)
 * @param eps epsilon threshold which decides if a value is considered zero
 */
Eigen::Quaterniond getAttiErr(const Eigen::Vector3d src, const Eigen::Vector3d dst, const double eps = double(1e-5))
{
	// Note: Eigen::Quaternion has two initialization methods: Eigen::Quaternion(Eigen::Vector4d(x, y, z, w)) and Eigen::Quaternion(w, x, y, z)
	Eigen::Vector4d q(0,0,0,0);
	Eigen::Vector3d cr = src.cross(dst);
	const float dt = src.dot(dst);

	if (cr.norm() < eps && dt < 0) {
		// handle corner cases with 180 degree rotations
		// if the two vectors are parallel, cross product is zero
		// if they point opposite, the dot product is negative
		cr = src.array().abs();

		if (cr(0) < cr(1)) {
			if (cr(0) < cr(2)) {
				cr = Eigen::Vector3d(1, 0, 0);

			} else {
				cr = Eigen::Vector3d(0, 0, 1);
			}

		} else {
			if (cr(1) < cr(2)) {
				cr = Eigen::Vector3d(0, 1, 0);

			} else {
				cr = Eigen::Vector3d(0, 0, 1);
			}
		}

		q(0) = double(0);
		cr = src.cross(cr);

	} else {
		// normal case, do half-way quaternion solution
		q(0) = dt + sqrt((src.squaredNorm() * dst.squaredNorm()));
	}

	q(1) = cr(0);
	q(2) = cr(1);
	q(3) = cr(2);
	q.normalize();
	return Eigen::Quaterniond(q(0),q(1),q(2),q(3));
}
class Px4AttitudeController{
private:
    /* data */
    Eigen::Quaterniond _attitude_setpoint_q; ///< latest known attitude setpoint e.g. from position control
	float _yaw_w{0.f}; ///< yaw weight [0,1] to deprioritize caompared to roll and pitch
	Eigen::Vector3d _proportional_gain;
	Eigen::Vector3d _rate_limit;
	float _yawspeed_setpoint{0.f}; ///< latest known yawspeed feed-forward setpoint

	Eigen::Vector3d _atti_target;
	Eigen::Vector3d _atti_now;



public:
    Px4AttitudeController(/* args */);
    ~Px4AttitudeController();
	void set_pid_params(Eigen::Vector3d p_gain)
	{
		_proportional_gain = p_gain;
	};
	Eigen::Vector3d update(const Eigen::Vector4d &q_sp,const Eigen::Vector4d &q)
	{
		// arg : set ,get
		// q_seq : w,x,y,z
		return update(Eigen::Quaterniond(q(0),q(1),q(2),q(3)),Eigen::Quaterniond(q_sp(0),q_sp(1),q_sp(2),q_sp(3)));
	}

    Eigen::Vector3d update(const Eigen::Quaterniond &q,const Eigen::Quaterniond &attitude_setpoint_q);
    // Eigen::Vector3d update_temp(const Eigen::Quaterniond &q,const Eigen::Quaterniond &attitude_setpoint_q);
	Eigen::Vector3d get_atti_target(){return _atti_target;};
	Eigen::Vector3d get_atti_now(){return _atti_now;};
};

Px4AttitudeController::Px4AttitudeController(/* args */)
{
	_rate_limit << 1600/57.3,1600/57.3,1000/57.3;
	set_pid_params(Eigen::Vector3d(8.0, 8.0, 1.0));
	_yaw_w = 0.400;
	// compensate for the effect of the yaw weight rescaling the output
	if (_yaw_w > 1e-4f) {
		_proportional_gain(2) /= _yaw_w;
	}
	_yawspeed_setpoint = 0;
}

Px4AttitudeController::~Px4AttitudeController()
{
}
// canonical form
Eigen::Quaterniond eigen_canonical(Eigen::Quaterniond q)
{
	Eigen::Quaterniond q_ret;
	double flag = MyMath::sign(q.w()); 

	q_ret.w() = q.w() * flag;
	q_ret.x() = q.x() * flag;
	q_ret.y() = q.y() * flag;
	q_ret.z() = q.z() * flag;
	return q_ret;
}
Eigen::Vector3d Px4AttitudeController::update(const Eigen::Quaterniond &q,const Eigen::Quaterniond &attitude_setpoint_q)
{
	// The coordinate systems of the desired and feedback are both in the world frame
    Eigen::Quaterniond qd = attitude_setpoint_q;
    Eigen::Vector3d e_z = q.toRotationMatrix().col(2);
    Eigen::Vector3d e_z_d = qd.toRotationMatrix().col(2);

    Eigen::Quaterniond qd_red = getAttiErr(e_z,e_z_d);

	if (fabsf(qd_red.x()) > (1.f - 1e-5f) || fabsf(qd_red.y()) > (1.f - 1e-5f)) {
        // 当 四元数中的 x或y项非常接近1时， 当前推力朝向与目标推力朝向完全相反，此时Full attitude control 不会产生任何Yaw角输入
        // 而是直接采用Roll和Pitch的组合，从而获得正确的Yaw.忽略这种情况仍然是安全和稳定的。
		// In the infinitesimal corner case where the vehicle and thrust have the completely opposite direction,
		// full attitude control anyways generates no yaw input and directly takes the combination of
		// roll and pitch leading to the correct desired yaw. Ignoring this case would still be totally safe and stable.
		qd_red = qd; 

	} else {
		// transform rotation from current to desired thrust vector into a world frame reduced desired attitude
		qd_red *= q;
	}

	// mix full and reduced desired attitude
	Eigen::Quaterniond q_mix = qd_red.inverse() * qd;
	q_mix = eigen_canonical(q_mix); // 
	// catch numerical problems with the domain of acosf and asinf
	q_mix.w() = MyMath::constrain(q_mix.w(), (double)-1.f, (double)1.f);
	q_mix.z() = MyMath::constrain(q_mix.z(), (double)-1.f, (double)1.f);
	qd = qd_red * Eigen::Quaterniond(cosf(_yaw_w * acosf(q_mix.w())), 0, 0, sinf(_yaw_w * asinf(q_mix.z())));
	
	// qd = qd_red;
	// quaternion attitude control law, qe is rotation from q to qd
	Eigen::Quaterniond qe = q.inverse() * qd;

	// using sin(alpha/2) scaled rotation axis as attitude error (see quaternion definition by axis angle)
	// also taking care of the antipodal unit quaternion ambiguity
    
	qe = eigen_canonical(qe);
	Eigen::Vector3d eq (qe.x(),qe.y(),qe.z());
	eq *= 2.f;

	// calculate angular rates setpoint
	Eigen::Vector3d rate_setpoint = eq.cwiseProduct(_proportional_gain);

	// Feed forward the yaw setpoint rate.
	// yawspeed_setpoint is the feed forward commanded rotation around the world z-axis,
	// but we need to apply it in the body frame (because _rates_sp is expressed in the body frame).
	// Therefore we infer the world z-axis (expressed in the body frame) by taking the last column of R.transposed (== q.inversed)
	// and multiply it by the yaw setpoint rate (yawspeed_setpoint).
	// This yields a vector representing the commanded rotatation around the world z-axis expressed in the body frame
	// such that it can be added to the rates setpoint.
    rate_setpoint += q.inverse().toRotationMatrix().col(2)* _yawspeed_setpoint;

	// limit rates
	for (int i = 0; i < 3; i++) {
		rate_setpoint(i) = MyMath::constrain(rate_setpoint(i), -_rate_limit(i), _rate_limit(i));
	}

	return rate_setpoint; //(body frame)
}
#endif