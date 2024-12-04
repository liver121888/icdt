from setuptools import setup

package_name = 'whisper_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'sounddevice', 'numpy', 'pynput'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Whisper audio transcription ROS 2 node',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_transcription = whisper_ros.task_transcription:main',
        ],
    },
)
