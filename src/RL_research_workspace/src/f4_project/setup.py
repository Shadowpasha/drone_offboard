from setuptools import setup, find_packages
import os
import sys
from glob import glob

# Shim to translate --editable (passed by some versions of colcon) to develop
if '--editable' in sys.argv:
    sys.argv.remove('--editable')
    if 'develop' not in sys.argv:
        sys.argv.append('develop')

package_name = 'f4_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'f4_project.TD3': ['models/*'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, "worlds"), glob('worlds/empty.world')),
        (os.path.join('share', package_name, "worlds"), glob('worlds/empty_sped_up.world')),
        (os.path.join('share', package_name, "launch"), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='anas',
    maintainer_email='anas@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'reset_odom = f4_project.reset_odom_mobile:main',
            'test_td3 = f4_project.TD3.test_td3:main',
        ],
    },
)
