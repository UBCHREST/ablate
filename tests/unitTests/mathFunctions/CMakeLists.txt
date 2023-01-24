target_sources(unitTests
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
        )

add_subdirectory(geom)