target_sources(ablateLibrary
        PRIVATE
        completeSublimation.cpp
        oneDimensionHeatTransfer.cpp
        temperatureSublimation.cpp

        PUBLIC
        sublimationModel.hpp
        completeSublimation.hpp
        oneDimensionHeatTransfer.hpp
        temperatureSublimation.hpp
)