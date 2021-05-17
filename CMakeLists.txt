
cmake_minimum_required(VERSION 3.4)

project(test)

find_package(robotskinlib REQUIRED)
##########################################################

add_executable(test test_opengl.cpp )
target_link_libraries(test robotskinlib::robotskinlib -lpthread)
