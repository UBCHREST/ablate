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
        )

add_subdirectory(boundaryConditions)