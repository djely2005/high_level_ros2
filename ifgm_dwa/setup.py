from setuptools import find_packages, setup

package_name = 'ifgm_dwa'

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
    maintainer='elyass',
    maintainer_email='elyass162005@gmail.com',
    description='Implementation of a combined algorithm of Improved Follow the Gap and Dynamic Window Approach to control the autonomous car for the CoVAPSy challenge.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
