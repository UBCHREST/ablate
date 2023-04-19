target_sources(unitTests
        PRIVATE
        dmPlexTests.cpp
        fieldDescriptionTests.cpp
        dynamicRangeTests.cpp
        reverseRangeTests.cpp

        PUBLIC
        mockField.hpp
        )

add_subdirectory(modifiers)
add_subdirectory(RBF)
