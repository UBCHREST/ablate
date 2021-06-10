target_sources(ablateLibrary
        PUBLIC
        flow.cpp
        flow.hpp
        flowFieldDescriptor.hpp
        lowMachFlow.hpp
        lowMachFlow.cpp
        incompressibleFlow.hpp
        incompressibleFlow.cpp
        compressibleFlow.cpp
        compressibleFlow.hpp
        )

add_subdirectory(boundaryConditions)
