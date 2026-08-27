#include "Px4AttitudeController.hpp"
#include "Px4RateController.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "SimplePositionController.hpp"
#include "Px4Mixer.hpp"

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(pyControl, m)
{
    m.doc() = R"pbdoc(
        Quadrotor Controller Utils
        -----------------------
    )pbdoc";

    py::class_<Px4AttitudeController>(m, "AttiControl")
        .def(py::init<>(), "Quaternion nonlinear attitude control, output in body coordinates")
        .def("set_pid_params", &Px4AttitudeController::set_pid_params, py::arg("p_gains").none())
        .def("update", py::overload_cast<const Eigen::Vector4d &, const Eigen::Vector4d &>(&Px4AttitudeController::update),
             py::arg("q_sp").none(), py::arg("q").none());

    py::class_<Px4RateController>(m, "RateControl")
        .def(py::init<>())
        .def("set_q_world", &Px4RateController::set_q_world, py::arg("q_world").none())
        .def("set_pid_params", &Px4RateController::set_pid_params, py::arg("p_gains").none(), py::arg("i_gains").none(), py::arg("d_gains").none())
        .def("update", py::overload_cast<const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &, const float>(&Px4RateController::update),
             py::arg("rate_sp").none(), py::arg("rate").none(), py::arg("angular_accel").none(), py::arg("dt").none());

    py::class_<Px4Mixer>(m, "Mixer")
        .def(py::init<>())
        .def("update", py::overload_cast<const Eigen::Vector4d &>(&Px4Mixer::update),
             py::arg("torque").none());

    py::class_<SimplePositionController>(m, "PosControl")
        .def(py::init<>())
        .def("get_hover_thrust", &SimplePositionController::get_hover_thrust)
        .def("set_pid_params", &SimplePositionController::set_pid_params, py::arg("pos_gains").none(), py::arg("vel_gains").none())
        .def("set_status", &SimplePositionController::set_status, py::arg("pos").none(), py::arg("vel").none(), py::arg("angular_velocity").none(), py::arg("q").none(), py::arg("dt").none())
        .def("update", py::overload_cast<const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &, const double>(&SimplePositionController::update),
             py::arg("pos_sp").none(), py::arg("vel_sp").none(), py::arg("acc_sp").none(), py::arg("yaw_sp").none());

    ;
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}