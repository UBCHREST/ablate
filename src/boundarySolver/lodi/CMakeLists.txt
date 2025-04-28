target_sources(ablateLibrary
        PRIVATE
        lodiBoundary.cpp
        isothermalWall.cpp
        openBoundary.cpp
        inlet.cpp
        massFluxInlet.cpp

        PUBLIC
        lodiBoundary.hpp
        isothermalWall.hpp
        openBoundary.hpp
        inlet.hpp
        massFluxInlet.hpp
        )

