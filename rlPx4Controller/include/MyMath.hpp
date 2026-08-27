#ifndef __MY_MATH_HPP_
#define __MY_MATH_HPP_
#include <Eigen/Eigen>
namespace MyMath
{
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
			Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX())
			);
}

template<typename _Tp>
constexpr _Tp min(_Tp a, _Tp b)
{
	return (a < b) ? a : b;
}

template<typename _Tp>
constexpr _Tp min(_Tp a, _Tp b, _Tp c)
{
	return min(min(a, b), c);
}

template<typename _Tp>
constexpr _Tp max(_Tp a, _Tp b)
{
	return (a > b) ? a : b;
}

template<typename _Tp>
constexpr _Tp max(_Tp a, _Tp b, _Tp c)
{
	return max(max(a, b), c);
}

/**
 * Type-safe sign/signum function
 *
 * @param[in] val Number to take the sign from
 * @return -1 if val < 0, 0 if val == 0, 1 if val > 0
 */
template<typename T>
int sign(T val)
{
	return (T(0) < val) - (val < T(0));
}

template<typename _Tp>
constexpr _Tp constrain(_Tp val, _Tp min_val, _Tp max_val)
{
	return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

template<typename T>
constexpr T radians(T degrees)
{
	return degrees * (static_cast<T>(M_PI) / static_cast<T>(180));
}
}



#endif