import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'icdt_wrapper'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liver',
    maintainer_email='liver121888@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wrapper = icdt_wrapper.wrapper:main',
            'talker = icdt_wrapper.publisher_member_function:main',
            'listener = icdt_wrapper.subscriber_member_function:main',
        ],
    },
)
