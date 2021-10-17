target_sources(ablateLibrary
        PUBLIC
        finiteElement.hpp
        finiteElement.cpp
        lowMachFlow.hpp
        lowMachFlow.cpp
        incompressibleFlow.hpp
        incompressibleFlow.cpp
        )

add_subdirectory(boundaryConditions)