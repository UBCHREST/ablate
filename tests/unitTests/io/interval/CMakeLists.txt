target_sources(unitTests
        PRIVATE
        fixedIntervalTests.cpp
        simulationTimeIntervalTests.cpp
        wallTimeIntervalTests.cpp
        delayIntervalTests.cpp
        equationIntervalTests.cpp

        PUBLIC
        mockInterval.hpp
        )
