function(add_unit_test executable)
    file(GLOB_RECURSE srcs CONFIGURE_DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        )
    add_executable(${executable}
        ${srcs}
        )
    target_include_directories(${executable} SYSTEM PRIVATE
        ${PROJECT_SOURCE_DIR}/third-party/googletest/googletest/include
        ${CUDA_INCLUDE_DIRS}
        )
    target_link_libraries(${executable} PRIVATE
        ${_C_LIBRARIES}
        GTest::gtest_main
        )
endfunction()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
add_subdirectory(test)