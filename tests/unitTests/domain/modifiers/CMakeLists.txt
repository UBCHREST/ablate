target_sources(ablateUnitTestLibrary
        PRIVATE
        meshMapperTests.cpp
        translateTests.cpp
        onePointClusteringMapperTests.cpp
        edgeClusteringMapperTests.cpp
        twoPointClusteringMapperTests.cpp

        PUBLIC
        meshMapperTestFixture.hpp
        )
