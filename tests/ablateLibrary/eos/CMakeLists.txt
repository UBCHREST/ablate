target_sources(libraryTests
        PRIVATE
        mockEOS.hpp
        perfectGasTests.cpp
        tChemTests.cpp
        stiffenedGasTests.cpp
        )

add_subdirectory(transport)
add_subdirectory(radiationProperties)