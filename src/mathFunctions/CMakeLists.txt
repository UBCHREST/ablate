target_sources(ablateLibrary
        PRIVATE
        simpleFormula.cpp
        functionWrapper.cpp
        functionPointer.cpp
        fieldFunction.cpp
        constantValue.cpp
        parsedSeries.cpp
        linearTable.cpp
        formula.cpp

        PUBLIC
        simpleFormula.hpp
        functionWrapper.hpp
        functionPointer.hpp
        mathFunction.hpp
        functionFactory.hpp
        fieldFunction.hpp
        constantValue.hpp
        parsedSeries.hpp
        linearTable.hpp
        formula.hpp
        )

add_subdirectory(geom)