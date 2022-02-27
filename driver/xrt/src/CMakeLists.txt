# MPI
include_directories(${MPI_CXX_INCLUDE_PATH})
link_libraries(${MPI_CXX_LIBRARIES} uuid)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${MPI_CXX_LINK_FLAGS})

# Json
get_target_property(JSON_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${JSON_INC_PATH})

add_executable(dacusr main.cpp accl.cpp)
# add_executable(bo bo.cpp)
# add_executable(m2m m2m.cpp)
