from setuptools import setup
import os
from glob import glob

package_name = 'f4_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, "worlds"), glob('worlds/empty.world')),
        (os.path.join('share', package_name, "worlds"), glob('worlds/empty_sped_up.world')),
        (os.path.join('share', package_name, "launch"), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anas',
    maintainer_email='anas@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'reset_odom = f4_project.reset_odom_mobile:main',   
        ],
    },
)
