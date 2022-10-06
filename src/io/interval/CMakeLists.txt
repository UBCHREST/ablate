target_sources(ablateLibrary
        PRIVATE
        fixedInterval.cpp
        wallTimeInterval.cpp
        simulationTimeInterval.cpp
        delayInterval.cpp

        PUBLIC
        interval.hpp
        fixedInterval.hpp
        simulationTimeInterval.hpp
        wallTimeInterval.hpp
        delayInterval.hpp
        )
