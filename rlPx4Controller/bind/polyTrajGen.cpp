#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "polynomial_traj.h"
#include <iostream>

namespace py = pybind11;
PolynomialTraj gl_traj;

class PolyTrajGen
{
private:
    /* data */
    Eigen::MatrixXd _pos_matrix;
    Eigen::Vector3d _start_vel, _end_vel, _start_acc, _end_acc;
    Eigen::VectorXd _time_time;

public:
    PolyTrajGen(Eigen::MatrixXd pos_matrix, Eigen::VectorXd dt_vec,
        Eigen::Vector3d start_vel = Eigen::Vector3d(0.f, 0.f, 0.f), Eigen::Vector3d start_acc = Eigen::Vector3d(0.f, 0.f, 0.f),
        Eigen::Vector3d end_vel = Eigen::Vector3d(0.f, 0.f, 0.f), Eigen::Vector3d end_acc = Eigen::Vector3d(0.f, 0.f, 0.f));
    void reset_traj(Eigen::MatrixXd pos_matrix, Eigen::VectorXd dt_vec,
        Eigen::Vector3d start_vel = Eigen::Vector3d(0.f, 0.f, 0.f), Eigen::Vector3d start_acc = Eigen::Vector3d(0.f, 0.f, 0.f),
        Eigen::Vector3d end_vel = Eigen::Vector3d(0.f, 0.f, 0.f), Eigen::Vector3d end_acc = Eigen::Vector3d(0.f, 0.f, 0.f));
    Eigen::Vector3d sample(double t);
    Eigen::Vector3d sample_vel(double t);
    Eigen::Vector3d sample_acc(double t);

    ~PolyTrajGen();
};

PolyTrajGen::PolyTrajGen(Eigen::MatrixXd pos_matrix, Eigen::VectorXd dt_vec,
        Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
        Eigen::Vector3d end_vel, Eigen::Vector3d end_acc)
{
    // pos_matrix 3*n
    if (pos_matrix.cols() != 3) {
        throw py::value_error("pos_matrix.rows() != 3");
    }
    if (dt_vec.size() != pos_matrix.rows() - 1) {
        throw py::value_error("dt_vec.size() != pos_matrix.cols() - 1");
    }

    _pos_matrix.resize(pos_matrix.cols(),pos_matrix.rows());
    _pos_matrix = pos_matrix.transpose();
    // pos <<  0,13,13,10,9,0,
    //         0,2,-8,-8, -2,0,
    //         1.2,1,1,1, 1,1.2;

    _start_vel = start_vel;
    _end_vel = end_vel;
    _start_acc = start_acc;
    _end_acc = end_acc;

    _time_time.resize(dt_vec.size());
    // time_time << 10,5,3,8,15;
    _time_time = dt_vec.transpose();


    gl_traj = PolynomialTraj::minSnapTraj(_pos_matrix, _start_vel, _end_vel, _start_acc, _end_acc, _time_time);
    gl_traj.init();

}

void PolyTrajGen::reset_traj(Eigen::MatrixXd pos_matrix, Eigen::VectorXd dt_vec,
        Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
        Eigen::Vector3d end_vel, Eigen::Vector3d end_acc)
{
    assert(pos_matrix.rows() == 3);
    assert(dt_vec.size() == pos_matrix.cols() - 1);

    _pos_matrix.resize(pos_matrix.cols(),pos_matrix.rows());
    _pos_matrix = pos_matrix.transpose();

    _start_vel = start_vel;
    _end_vel = end_vel;
    _start_acc = start_acc;
    _end_acc = end_acc;

    _time_time.resize(dt_vec.size());
    _time_time = dt_vec.transpose();
    gl_traj = PolynomialTraj::minSnapTraj(_pos_matrix, _start_vel, _end_vel, _start_acc, _end_acc, _time_time);
    gl_traj.init();

}
Eigen::Vector3d PolyTrajGen::sample(double t)
{
    double global_duration = gl_traj.getTimeSum();

    if(t < global_duration)
        return gl_traj.evaluate(t);
    else
        return gl_traj.evaluate(global_duration - 0.1);
}
Eigen::Vector3d PolyTrajGen::sample_vel(double t)
{
    double global_duration = gl_traj.getTimeSum();

    if(t < global_duration)
        return gl_traj.evaluateVel(t);
    else
        return gl_traj.evaluateVel(global_duration - 0.1);
}
Eigen::Vector3d PolyTrajGen::sample_acc(double t)
{
    double global_duration = gl_traj.getTimeSum();

    if(t < global_duration)
        return gl_traj.evaluateAcc(t);
    else
        return gl_traj.evaluateAcc(global_duration - 0.1);
}

PolyTrajGen::~PolyTrajGen()
{
}
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(polyTrajGen, m) {
    m.doc() = R"pbdoc(
        polynomial_traj
        -----------------------
    )pbdoc";

    py::class_<PolyTrajGen>(m, "PolyTrajGen")
    .def(py::init<Eigen::MatrixXd,Eigen::VectorXd,Eigen::Vector3d,Eigen::Vector3d,Eigen::Vector3d,Eigen::Vector3d>())
    .def("sample",&PolyTrajGen::sample)
    .def("reset_traj",&PolyTrajGen::reset_traj)
    .def("sample_vel",&PolyTrajGen::sample_vel)
    .def("sample_acc",&PolyTrajGen::sample_acc)
    ;

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}