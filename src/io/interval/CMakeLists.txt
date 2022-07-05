target_sources(ablateLibrary
        PRIVATE
        fixedInterval.cpp
        wallTimeInterval.cpp
        simulationTimeInterval.cpp

        PUBLIC
        interval.hpp
        fixedInterval.hpp
        simulationTimeInterval.hpp
        wallTimeInterval.hpp
        )
