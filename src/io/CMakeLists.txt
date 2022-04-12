target_sources(ablateLibrary
        PUBLIC
        serializable.hpp
        serializer.hpp
        hdf5Serializer.hpp
        hdf5Serializer.cpp
        hdf5MultiFileSerializer.hpp
        hdf5MultiFileSerializer.cpp
        )

add_subdirectory(interval)
