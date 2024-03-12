target_sources(ablateUnitTestLibrary
        PRIVATE
        dmPlexTests.cpp
        fieldDescriptionTests.cpp
        dynamicRangeTests.cpp
        reverseRangeTests.cpp
        hdf5InitializerTests.cpp
        fieldAccessorTests.cpp

        PUBLIC
        mockField.hpp
        )

add_subdirectory(modifiers)
add_subdirectory(RBF)
add_subdirectory(descriptions)
