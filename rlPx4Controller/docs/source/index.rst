Welcome to rlPx4Controller's documentation!
===========================================

**rlPx4Controller** is a quadcopter control library implemented in C++ and provides a python interface through pybind11. The implementation of the controller is consistent with the Px4 flight control to ensure the consistency of sim2real.

4 modules have been implemented so far

- pyControl: Single aircraft controller
- pyParallelControl: Multi-aircraft controller for reinforcement learning
- PolyTrajGen: Fifth degree polynomial trajectory generation
- Lemniscate:  Figure-eight curve generation

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   dynamic
   sim_support
   api