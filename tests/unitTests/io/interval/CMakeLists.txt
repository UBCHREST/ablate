target_sources(ablateUnitTestLibrary
        PRIVATE
        fixedIntervalTests.cpp
        simulationTimeIntervalTests.cpp
        wallTimeIntervalTests.cpp
        delayIntervalTests.cpp
        equationIntervalTests.cpp

        PUBLIC
        mockInterval.hpp
        )
