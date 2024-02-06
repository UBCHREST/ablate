target_sources(ablateLibrary
        PRIVATE
        tracer.cpp
        inertial.cpp
        arbitraryEulerianSource.cpp

        PUBLIC
        process.hpp
        tracer.hpp
        inertial.hpp
        coupledProcess.hpp
        arbitraryEulerianSource.hpp
        )

add_subdirectory(drag)