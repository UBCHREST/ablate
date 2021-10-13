# Setup the ablate core
add_library(ablateCore "")

# Include PETSc and MPI
target_link_libraries(ablateCore PUBLIC PkgConfig::PETSc)

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