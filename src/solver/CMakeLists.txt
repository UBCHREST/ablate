target_sources(ablateLibrary
        PRIVATE
        timeStepper.cpp
        solver.cpp
        cellSolver.cpp
        adaptPhysics.cpp
        adaptPhysicsConstrained.cpp
        steadyStateStepper.cpp

        PUBLIC
        timeStepper.hpp
        solver.hpp
        iFunction.hpp
        rhsFunction.hpp
        boundaryFunction.hpp
        cellSolver.hpp
        adaptPhysics.hpp
        physicsTimeStepFunction.hpp
        steadyStateStepper.hpp
        )

add_subdirectory(criteria)