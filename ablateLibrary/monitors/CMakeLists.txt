target_sources(ablateLibrary
        PUBLIC
        viewable.hpp
        monitor.hpp
        hdf5Monitor.hpp
        hdf5Monitor.cpp
        fieldErrorMonitor.hpp
        fieldErrorMonitor.cpp
        solutionErrorMonitor.hpp
        solutionErrorMonitor.cpp
        timeStepMonitor.hpp
        timeStepMonitor.cpp
        )

add_subdirectory(logs)