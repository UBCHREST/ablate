target_sources(libraryTests
        PRIVATE
        simpleFormulaTests.cpp
        functionWrapperTests.cpp
        functionPointerTests.cpp
        functionFactoryTests.cpp
        constantValueTests.cpp
        parsedSeriesTests.cpp
        linearInterpolatorTests.cpp
        formulaTests.cpp
        )

add_subdirectory(geom)