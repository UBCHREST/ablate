
# Setup the framework library
target_sources(testingResources
        PRIVATE
        fileAssert.cpp
        stdOutAssert.cpp
        textFileAssert.cpp

        PUBLIC
        assert.hpp
        perRankAssert.hpp
        postRunAssert.hpp
        stdOutAssert.hpp
        textFileAssert.hpp
)