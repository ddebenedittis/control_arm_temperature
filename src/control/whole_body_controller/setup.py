from setuptools import find_packages, setup

package_name = 'whole_body_controller'

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
    maintainer='davide',
    maintainer_email='davide@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wbc_node = whole_body_controller.wbc_node:main',
            'wbc_leg_node = whole_body_controller.wbc_leg_node:main',
            'temperature_node = whole_body_controller.temperature_node:main',
        ],
    },
)
