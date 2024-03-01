target_sources(ablateLibrary
        PRIVATE
        axisymmetric.cpp
        generatedAxis.cpp
        specifiedAxis.cpp

        PUBLIC
        meshDescription.hpp
        axisymmetric.hpp
        axisDescription.hpp
        generatedAxis.hpp
        specifiedAxis.hpp
        )
