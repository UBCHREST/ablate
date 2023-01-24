target_sources(ablateLibrary
        PRIVATE
        radiation.cpp
        volumeRadiation.cpp
        surfaceRadiation.cpp

        PUBLIC
        radiation.hpp
        volumeRadiation.hpp
        surfaceRadiation.hpp
        )