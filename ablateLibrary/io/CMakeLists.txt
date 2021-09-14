target_sources(ablateLibrary
        PUBLIC
        serializable.hpp
        serializer.hpp
        hdf5Serializer.hpp
        hdf5Serializer.cpp
        )

add_subdirectory(interval)
