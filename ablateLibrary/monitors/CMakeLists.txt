target_sources(ablateLibrary
        PUBLIC
        monitor.hpp
        hdf5Output.hpp
        hdf5Output.cpp
        )

add_subdirectory(flow)
add_subdirectory(particles)