target_sources(ablateLibrary
        PRIVATE
        tracer.cpp
        inertial.cpp

        PUBLIC
        process.hpp
        tracer.hpp
        inertial.hpp
        )

add_subdirectory(drag)