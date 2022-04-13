target_sources(ablateLibrary
        PUBLIC
        finiteElementSolver.hpp
        finiteElementSolver.cpp
        lowMachFlowSolver.hpp
        lowMachFlowSolver.cpp
        incompressibleFlowSolver.hpp
        incompressibleFlowSolver.cpp
        lowMachFlowFields.hpp
        lowMachFlowFields.cpp
        incompressibleFlow.h
        incompressibleFlow.c
        lowMachFlow.h
        lowMachFlow.c
        )

add_subdirectory(boundaryConditions)