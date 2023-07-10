target_sources(ablateLibrary
        PRIVATE
        sublimation.cpp
        arbitrarySource.cpp
        logLawBoundary.cpp

        PUBLIC
        sublimation.hpp
        arbitrarySource.hpp
        logLawBoundary.hpp
)

add_subdirectory(subModels)