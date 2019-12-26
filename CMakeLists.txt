cmake_minimum_required(VERSION 3.1)

project(main C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
option(DEBUG "output debug information")


find_package(SDL2 REQUIRED)
find_library(SDL2_LIBRARY REQUIRED)

if(DEBUG)
    get_cmake_property(_variableNames VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif()

set(PACKAGE_DIRS ${SDL2_INCLUDE_DIRS} )
set(PACKAGE_LIBS ${SDL2_LIBRARIES} )


include_directories(${PACKAGE_DIRS})
file(GLOB_RECURSE SRC_HDR include/*.h)
file(GLOB_RECURSE SRC_CPP src/*.cpp)
add_executable(main main.cpp ${SRC_HDR}
        ${SRC_CPP})
target_link_libraries(main ${PACKAGE_LIBS})
target_link_libraries(main ale
        

add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/roms $<TARGET_FILE_DIR:main>/roms)