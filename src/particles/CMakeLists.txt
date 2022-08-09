target_sources(ablateLibrary
        PRIVATE
        particles.cpp
        tracer.cpp
        inertial.cpp
        particleSolver.hpp

        PUBLIC
        particles.hpp
        tracer.hpp
        inertial.hpp
        particleField.hpp
        particleSolver.cpp
        )

add_subdirectory(initializers)
add_subdirectory(drag)
add_subdirectory(processes)
