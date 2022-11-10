target_sources(unitTests
        PRIVATE
        meshMapperTests.cpp
        translateTests.cpp
        onePointClusteringMapperTests.cpp
        edgeClusteringMapperTests.cpp

        PUBLIC
        meshMapperTestFixture.hpp
        )
