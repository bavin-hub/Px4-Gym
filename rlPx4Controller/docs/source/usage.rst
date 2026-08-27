Usage
=====

.. _installation:

Installation
------------

To use rlPx4Controller , first install it using pip:

.. code-block:: console

   # Install Eigen. The recommended version is 3.3.7
   sudo apt install libeigen3-dev
   pip install pybind11 pybind11_stubgen
   # install rlPx4Controller
   pip install -e .
   bash stubgen.sh



Single Quadcopter Test
----------------------

.. To retrieve a list of random ingredients,
.. you can use the ``lumache.get_random_ingredients()`` function:

.. .. autofunction:: lumache.get_random_ingredients

.. The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
.. or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
.. will raise an exception.

.. .. autoexception:: lumache.InvalidKindError

For example:

.. code-block:: python

   from rlPx4Controller.pyControl import PosControl,AttiControl,RateControl,Mixer

   controller.pos_ctl.set_status(pos_world,velocity_world,angular_velocity_world,rot_quat,current_time-last_rate_control_time)
   atti_thrust_sp = controller.pos_ctl.update(exp_pos,np.array([0,0,0]),np.array([0,0,0]),float(yaw))

   print("hover_thrust {} ".format(controller.pos_ctl.get_hover_thrust()))

   rate_sp = controller.atti_ctl.update(atti_thrust_sp[:4],rot_quat)

   controller.rate_ctl.set_q_world(rot_quat)
   thrust_3 = controller.rate_ctl.update(rate_sp,angular_velocity_world,np.array([0,0,0]),current_time-last_rate_control_time)

   thrust = controller.mix_ctl.update(np.array([thrust_3[0],thrust_3[1],thrust_3[2],atti_thrust_sp[4]]))


.. >>> import lumache
.. >>> lumache.get_random_ingredients()
.. ['shells', 'gorgonzola', 'parsley']

Multi Quadcopter Test
----------------------
Since the python for loop is too slow, I chose to implement it in the `pyParallelControl` package.

For example:


.. code:: python

   from rlPx4Controller.pyParallelControl import ParallelPosControl

   parallel_rate_ctl = ParallelRateControl(envs_num)
   while True:
      parallel_rate_ctl.set_q_world(root_rot_cpu)
      cmd = parallel_rate_ctl.update(rate_sp,rate,dt)