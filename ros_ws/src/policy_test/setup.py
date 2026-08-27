from setuptools import find_packages, setup

package_name = 'policy_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bavin',
    maintainer_email='bavinsaravanan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "position_sim_controller = policy_test.position_controller:main",
            "velocity_sim_controller = policy_test.velocity_controller:main",
            "delta_attitude_sp = policy_test.starling_attitude_controller:main",
            "body_rates_sim_controller = policy_test.body_rates_controller:main",
            "actuator_sim_controller = policy_test.actuator_controller:main",
            "offb_control = policy_test.offb:main"
        ],
    },
)
