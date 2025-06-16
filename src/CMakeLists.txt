choreonoid_add_simple_controller(RttTankController RttTankController.cpp)
 target_link_libraries(RttTankController ${roscpp_LIBRARIES})

find_package(Torch REQUIRED)
choreonoid_add_simple_controller(Go2RosController Go2RosController.cpp)
target_link_libraries(Go2RosController ${roscpp_LIBRARIES} ${CHOREONOID_BASE_LIBRARIES} "${TORCH_LIBRARIES}")
set_property(TARGET Go2RosController PROPERTY CXX_STANDARD 17)