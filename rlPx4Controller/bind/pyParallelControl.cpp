#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "Px4AttitudeController.hpp"
#include "SimplePositionController.hpp"
#include "Px4Mixer.hpp"
#include "parallelRateController.hpp"
#include "parallelVelController.hpp"
#include "parallelAttiController.hpp"
#include "parallelPosController.hpp"
namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(pyParallelControl, m)
{
    m.doc() = R"pbdoc(
        Quadrotor Parallel Controller Utils
        -----------------------
    )pbdoc";

    py::class_<pyParallelRateControl>(m, "ParallelRateControl")
        .def(py::init<int>(), "batch calc")
        .def("set_q_world", &pyParallelRateControl::set_q_world, py::arg("q_world").none())
        .def("update", &pyParallelRateControl::update,
             py::arg("rate_sp").none(), py::arg("rate").none(), py::arg("dt").none());

    py::class_<pyParallelVelocityControl>(m, "ParallelVelControl")
        .def(py::init<int>(), "b")
        .def("set_status", &pyParallelVelocityControl::set_status, py::arg("pos").none(), py::arg("q_matrix").none(), py::arg("vel").none(), py::arg("ang_vel").none(), py::arg("dt").none())
        .def("update", &pyParallelVelocityControl::update, py::arg("actions").none());
    py::class_<pyParallelPositionControl>(m, "ParallelPosControl")
        .def(py::init<int>(), "b")
        .def("set_status", &pyParallelPositionControl::set_status, py::arg("pos").none(), py::arg("q_matrix").none(), py::arg("vel").none(), py::arg("ang_vel").none(), py::arg("dt").none())
        .def("update", &pyParallelPositionControl::update, py::arg("actions").none());
        
    py::class_<pyParallelAttiControl>(m, "ParallelAttiControl")
        .def(py::init<int>(), "b")
        .def("set_status", &pyParallelAttiControl::set_status, py::arg("pos").none(), py::arg("q_matrix").none(), py::arg("vel").none(), py::arg("ang_vel").none(), py::arg("dt").none())
        .def("update", &pyParallelAttiControl::update, py::arg("actions").none());

    ;
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}