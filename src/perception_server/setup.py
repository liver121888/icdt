from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='warra',
    maintainer_email='warrier.abhishek@gmail.com',
    description='Node to advertise detection services',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hello_world = perception_server.hello_world:main',
            'detection_publisher = perception_server.detection_publisher:main',
            'detection_service = perception_server.detection_service:main',
            'detection_simple_client = perception_server.detection_simple_client:main',
            'detection_interactive_client = perception_server.detection_interactive_client:main'
        ],
    },
)
