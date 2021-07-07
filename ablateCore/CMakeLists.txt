# Setup the ablate core
add_library(ablateCore "")

# Include PETSc and MPI
target_include_directories(ablateCore PUBLIC ${PETSc_INCLUDE_DIRS})
target_link_libraries(ablateCore PUBLIC ${PETSc_LIBRARIES})
target_link_directories(ablateCore PUBLIC ${PETSc_LIBRARY_DIRS})

# Allow public access to the header files in the directory
target_include_directories(ablateCore PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Include the code in any subdirectory
add_subdirectory(flow)

# Tag the version file
configure_file (
        "version.h.in"
        "version.h"
)
target_include_directories(ablateCore PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Add required warnings for the ablate core
ablate_default_target_compile_options_c(ablateCore)