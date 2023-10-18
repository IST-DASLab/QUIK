set(SRCS matmul.cpp)

if (CUDA_FOUND)
    list(APPEND SRCS matmul.cu)
endif ()


add_library(_C_LIBRARY_MATMUL STATIC "${SRCS}")
target_link_libraries(_C_LIBRARY_MATMUL PRIVATE ${TORCH_LIBRARIES} cusparseLt)
if (CUDA_FOUND)
    target_link_libraries(_C_LIBRARY_MATMUL PRIVATE nvidia::cutlass::cutlass nvidia::cutlass::tools::util)
endif ()

list(APPEND _C_LIBRARIES _C_LIBRARY_MATMUL)
set(_C_LIBRARIES "${_C_LIBRARIES}" PARENT_SCOPE)
