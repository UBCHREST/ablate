target_sources(ablateLibrary
        PUBLIC
        ../environment/runEnvironment.hpp
        ../environment/runEnvironment.cpp
        monitor.hpp
        hdf5Output.hpp
        hdf5Output.cpp
        )

add_subdirectory(flow)
add_subdirectory(particles)