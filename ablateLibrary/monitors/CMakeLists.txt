target_sources(ablateLibrary
        PUBLIC
        runEnvironment.hpp
        runEnvironment.cpp
        monitor.hpp
        hdf5Output.hpp
        hdf5Output.cpp
        )

add_subdirectory(flow)
add_subdirectory(particles)