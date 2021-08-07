CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(demo3)

find_package(OpenCV REQUIRED)

ADD_EXECUTABLE(demo3  main.cpp)

target_link_libraries(demo3 PRIVATE ${OpenCV_LIBS})
