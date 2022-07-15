target_sources(libraryTests
        PRIVATE
        mockEOS.hpp
        perfectGasTests.cpp
        stiffenedGasTests.cpp
        tChemTests.cpp
        )

add_subdirectory(transport)
add_subdirectory(radiationProperties)