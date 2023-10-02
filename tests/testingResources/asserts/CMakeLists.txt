
# Setup the framework library
target_sources(testingResources
        PRIVATE
        fileAssert.cpp
        stdOutAssert.cpp
        textFileAssert.cpp
        solutionVectorAssert.cpp

        PUBLIC
        assert.hpp
        stdOutAssert.hpp
        textFileAssert.hpp
        solutionVectorAssert.hpp
)