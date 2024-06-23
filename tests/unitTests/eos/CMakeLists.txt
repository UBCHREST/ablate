target_sources(ablateUnitTestLibrary
        PRIVATE
        mockEOS.hpp
        perfectGasTests.cpp
        stiffenedGasTests.cpp
        tChemTests.cpp
        chemTabTests.cpp
        twoPhaseTests.cpp
        tChemSootTests.cpp
        zerorkTest.cpp
        )

add_subdirectory(transport)
add_subdirectory(radiationProperties)
add_subdirectory(tChemSoot)
add_subdirectory(chemTab)