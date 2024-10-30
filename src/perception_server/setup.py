from setuptools import find_packages, setup

package_name = 'perception_server'

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
    maintainer='warra',
    maintainer_email='warrier.abhishek@gmail.com',
    description='Node to advertise detection services',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hello_world = perception_server.hello_world:main',
            'detection_service = perception_server.detection_service:main'
        ],
    },
)
