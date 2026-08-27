#include "Px4AttitudeController.hpp"
#include "Px4RateController.hpp"
#include "Px4Mixer.hpp"
#include <Eigen/Geometry>
#include <iostream>
#include "MyMath.hpp"
#include <Eigen/Eigen>
#include <Eigen/Geometry>



int main(){
    Px4AttitudeController attiControl;
    Px4RateController rateControl;
    Px4Mixer mixer;

    // while (true)
    // {
        Eigen::Vector3d euler_angles(-10/57.3, 10/57.3,.0);  // Roll, Pitch, Yaw angles
        // Eigen::Quaterniond quat = Eigen::AngleAxisd(euler_angles(0), Eigen::Vector3d::UnitX())
        //                                 * Eigen::AngleAxisd(euler_angles(1), Eigen::Vector3d::UnitY())
        //                                 * Eigen::AngleAxisd(euler_angles(2), Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond quat = Eigen::AngleAxisd(euler_angles(2), Eigen::Vector3d::UnitZ())
                                        * Eigen::AngleAxisd(euler_angles(1), Eigen::Vector3d::UnitY())
                                        * Eigen::AngleAxisd(euler_angles(0), Eigen::Vector3d::UnitX());
        std::cout<<quat.coeffs()<<std::endl;
	    std::cout<<"quat"<<quat.toRotationMatrix()<<std::endl;

        // Eigen::Quaterniond qd(rotation_matrix);
        Eigen::Vector3d rate_d =  attiControl.update_temp(Eigen::Quaterniond(1,0,0,0),quat);
        std::cout<<rate_d*57.3 <<std::endl;
        Eigen::Vector4d torque_d(0,0,0,0);


        Eigen::Quaterniond enu2ned = MyMath::quaternion_from_rpy(Eigen::Vector3d(M_PI,0,0));

        // static const Eigen::PermutationMatrix<3> NED_ENU_REFLECTION_XY(Eigen::Vector3i(1,0,2));
        // static const Eigen::DiagonalMatrix<double,3> NED_ENU_REFLECTION_Z(1,1,-1);
        // rate_d = NED_ENU_REFLECTION_XY * (NED_ENU_REFLECTION_Z * rate_d);
        torque_d.head<3>()=  rateControl.update(Eigen::Vector3d(0,0,0),rate_d,Eigen::Vector3d(0,0,0),0.01,false);
        std::cout<< "torque_d" << torque_d <<std::endl;
        std::cout<< mixer.update(torque_d)<<std::endl;


        std::cout<< "<<<<<<<<<<<<<<<<<<<<<<<test>>>>>>>>>>>>>>>>>>>>>>>" <<std::endl;

       Eigen::Vector3d euler_angles_test(-10/57.3, 10/57.3,.5/57.3);  // Roll, Pitch, Yaw angles
        // Eigen::Quaterniond quat = Eigen::AngleAxisd(euler_angles(0), Eigen::Vector3d::UnitX())
        //                                 * Eigen::AngleAxisd(euler_angles(1), Eigen::Vector3d::UnitY())
        //                                 * Eigen::AngleAxisd(euler_angles(2), Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond quat_test = Eigen::AngleAxisd(euler_angles_test(2), Eigen::Vector3d::UnitZ())
                                        * Eigen::AngleAxisd(euler_angles_test(1), Eigen::Vector3d::UnitY())
                                        * Eigen::AngleAxisd(euler_angles_test(0), Eigen::Vector3d::UnitX()); //Z-Y-X 的顺序
        Eigen::Matrix3d rotationMatrix = quat_test.toRotationMatrix();
        double roll, pitch, yaw;
        // roll = atan2(rotationMatrix(2, 1), rotationMatrix(2, 2));
        // pitch = asin(-rotationMatrix(2, 0));
        // yaw = atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));

	    std::cout << "euler_angles_test"  <<  rotationMatrix<< std::endl;

        // ZYX
        roll = atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));
        pitch = asin(-rotationMatrix(2, 0));
        yaw = atan2(rotationMatrix(2, 1), rotationMatrix(2, 2));
	    std::cout << "roll, pitch, yaw"  << " "<< roll << " "<< pitch  << " "<< yaw << std::endl;

        // XYZ  z-y-x的顺序
        roll = atan2(-rotationMatrix(1, 2), rotationMatrix(2, 2));
        pitch = asin(rotationMatrix(0, 2));
        yaw = atan2(-rotationMatrix(0, 1), rotationMatrix(0, 0));
        std::cout << "roll, pitch, yaw"  << " "<< roll << " "<< pitch  << " "<< yaw << std::endl;


        static const auto AIRCRAFT_BASELINK_Q = quaternion_from_rpy(M_PI, 0.0, 0.0);

        static const Eigen::Affine3d AIRCRAFT_BASELINK_AFFINE(AIRCRAFT_BASELINK_Q);
    
        std::cout<< "AIRCRAFT_BASELINK_AFFINE"  << AIRCRAFT_BASELINK_AFFINE.matrix()<< std::endl;

    // }
    return 0;
    
}