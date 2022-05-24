target_sources(libraryTests
        PRIVATE
        mockEOS.hpp
        perfectGasTests.cpp
        tChemV1Tests.cpp
        stiffenedGasTests.cpp
        )

add_subdirectory(transport)
add_subdirectory(radiationProperties)