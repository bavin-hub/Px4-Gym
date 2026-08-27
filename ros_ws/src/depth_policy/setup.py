from setuptools import find_packages, setup

package_name = 'depth_policy'

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
    maintainer_email='bavinsaravanan24@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'depth_attitude_sp = depth_policy.depth_attitude_sp:main',
            'depth_velocity_sp = depth_policy.depth_velocity_sp:main',
            'test_depth = depth_policy.test_depth:main',
        ],
    },
)
