target_sources(unitTests
        PRIVATE
        perfectGasTests.cpp
        stiffenedGasTests.cpp
        tChemTests.cpp
        chemTabTests.cpp
        twoPhaseTests.cpp
        tChemSootTests.cpp

        PUBLIC
        mockEOS.hpp
        chemTabTestFixture.hpp
)

add_subdirectory(transport)
add_subdirectory(radiationProperties)
add_subdirectory(tChemSoot)