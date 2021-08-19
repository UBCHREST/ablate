target_sources(libraryTests
        PRIVATE
        mockEOS.hpp
        perfectGasTests.cpp
        tChemTests.cpp
        )

add_subdirectory(transport)