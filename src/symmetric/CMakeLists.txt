set(SRCS symmetric.cpp)

if (CUDA_FOUND)
    list(APPEND SRCS symmetric.cu fused_dequantize.cu)
endif ()


add_library(_C_LIBRARY_SYMMETRIC STATIC "${SRCS}")
target_link_libraries(_C_LIBRARY_SYMMETRIC PRIVATE ${TORCH_LIBRARIES})
if (CUDA_FOUND)
    target_link_libraries(_C_LIBRARY_SYMMETRIC PRIVATE nvidia::cutlass::cutlass)
endif ()

list(APPEND _C_LIBRARIES _C_LIBRARY_SYMMETRIC)
set(_C_LIBRARIES "${_C_LIBRARIES}" PARENT_SCOPE)
