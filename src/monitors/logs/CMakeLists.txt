target_sources(ablateLibrary
        PRIVATE
        log.cpp
        csvLog.cpp
        streamLog.cpp
        fileLog.cpp
        stdOut.cpp

        PUBLIC
        log.hpp
        csvLog.hpp
        streamLog.hpp
        fileLog.hpp
        stdOut.hpp
        nullLog.hpp
        )