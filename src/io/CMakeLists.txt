target_sources(ablateLibrary
        PRIVATE
        hdf5Serializer.cpp
        hdf5MultiFileSerializer.cpp
        serializable.cpp

        PUBLIC
        serializable.hpp
        serializer.hpp
        hdf5Serializer.hpp
        hdf5MultiFileSerializer.hpp
        )

add_subdirectory(interval)
