target_sources(ablateLibrary
        PRIVATE
        finiteElementSolver.cpp
        lowMachFlowSolver.cpp
        incompressibleFlowSolver.cpp
        lowMachFlowFields.cpp
        incompressibleFlow.c
        lowMachFlow.c
        PUBLIC
        finiteElementSolver.hpp
        lowMachFlowSolver.hpp
        incompressibleFlowSolver.hpp
        lowMachFlowFields.hpp
        incompressibleFlow.h
        lowMachFlow.h
        )

add_subdirectory(boundaryConditions)