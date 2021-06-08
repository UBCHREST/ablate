target_sources(ablateLibrary
        PUBLIC
        particles.hpp
        particles.cpp
        particleFieldDescriptor.hpp
        particleFieldDescriptor.cpp
        tracer.hpp
        tracer.cpp
        inertial.hpp
        inertial.cpp
        )

add_subdirectory(initializers)
