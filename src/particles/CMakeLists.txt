target_sources(ablateLibrary
        PRIVATE
        particles.cpp
        tracer.cpp
        inertial.cpp
        particleSolver.cpp

        PUBLIC
        particles.hpp
        tracer.hpp
        inertial.hpp
        field.hpp
        particleSolver.hpp
        fieldDescription.hpp
        swarmData.hpp
        particleData.hpp
        rhsData.hpp
        eulerianData.hpp
        )

add_subdirectory(initializers)
add_subdirectory(drag)
add_subdirectory(processes)
