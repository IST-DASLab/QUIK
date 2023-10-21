set(TEST_UNITS "")
macro(add_qgpt_unit_test)
    file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    )
    list(APPEND TEST_UNITS ${srcs})
    set(TEST_UNITS "${TEST_UNITS}" PARENT_SCOPE)
endmacro()

add_subdirectory(linear_combination)
add_subdirectory(predicated_iterator)
add_subdirectory(gemm)


add_executable(quiktest run_test.cpp
        ${TEST_UNITS}
)
target_include_directories(quiktest SYSTEM PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/third-party/googletest/googletest/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(quiktest PRIVATE
        GTest::gtest_main
        nvidia::cutlass::cutlass
        nvidia::cutlass::tools::util
)
