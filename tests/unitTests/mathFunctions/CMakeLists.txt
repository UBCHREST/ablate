target_sources(ablateUnitTestLibrary
        PRIVATE
        simpleFormulaTests.cpp
        functionWrapperTests.cpp
        functionPointerTests.cpp
        functionFactoryTests.cpp
        constantValueTests.cpp
        parsedSeriesTests.cpp
        linearInterpolatorTests.cpp
        formulaTests.cpp
        linearTests.cpp
        peakTests.cpp
        linearFunctionTests.cpp
        nearestPointTests.cpp
        )

add_subdirectory(geom)