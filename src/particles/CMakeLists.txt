target_sources(ablateLibrary
        PRIVATE
        particles.cpp
        tracer.cpp
        inertial.cpp

        PUBLIC
        particles.hpp
        tracer.hpp
        inertial.hpp
        particleField.hpp
        )

add_subdirectory(initializers)
add_subdirectory(drag)
