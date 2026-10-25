cmake_minimum_required(VERSION 3.10)
project(cybership_viz)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  nodes/desired_force_stamped.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
