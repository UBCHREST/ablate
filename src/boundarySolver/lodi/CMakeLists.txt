target_sources(ablateLibrary
        PRIVATE
        lodiBoundary.cpp
        isothermalWall.cpp
        openBoundary.cpp
        inlet.cpp
        PUBLIC
        lodiBoundary.hpp
        isothermalWall.hpp
        openBoundary.hpp
        inlet.hpp
        )

