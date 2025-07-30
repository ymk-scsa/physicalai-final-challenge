from setuptools import find_packages, setup

package_name = 'final_challenge_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']), # ここに余分な ')' がないことを再々々確認してください
    data_files=[
        ('share/' + package_name, ['package.xml']),
        # ROS2がパッケージリソースを見つけられない問題を回避するため、以下の行をコメントアウト
        # ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Physical AI Final Challenge Package for Robot Movement and AI Learning',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = final_challenge_pkg.robot_controller:main',
        ],
    },
)
