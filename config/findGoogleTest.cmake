
IF(TARGET GTest::gtest AND TARGET GTest::gmock)
    message(STATUS "Found GTest::gtest and  GTest::gmock libaries")
ELSE()
    SET(INSTALL_GTEST OFF CACHE BOOL "Don't install gtest" FORCE)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG main
    )
    FetchContent_MakeAvailable(googletest)
ENDIF()