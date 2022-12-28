target_sources(unitTests
        PRIVATE
        mockEOS.hpp
        perfectGasTests.cpp
        stiffenedGasTests.cpp
        tChemTests.cpp
        chemTabTests.cpp
        twoPhaseTests.cpp
        )

add_subdirectory(transport)
add_subdirectory(radiationProperties)