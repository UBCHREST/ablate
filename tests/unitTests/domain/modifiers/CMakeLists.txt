target_sources(unitTests
        PRIVATE
        meshMapperTests.cpp
        translateTests.cpp
        onePointClusteringMapperTests.cpp
        edgeClusteringMapperTests.cpp
        twoPointClusteringMapperTests.cpp

        PUBLIC
        meshMapperTestFixture.hpp
        )
