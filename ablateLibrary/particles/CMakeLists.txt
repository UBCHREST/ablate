target_sources(ablateLibrary
        PUBLIC
        particles.hpp
        particles.cpp
        tracer.hpp
        tracer.cpp
        inertial.hpp
        inertial.cpp
        particleField.hpp
        )

add_subdirectory(initializers)
