target_sources(ablateLibrary
        PRIVATE
#        solidHeatTransfer.cpp
#        solidHeatTransferFactory.cpp
#        sublimationTemperature.cpp
        completeSublimation.cpp
        oneDimensionHeatTransfer.cpp

        PUBLIC
#        solidHeatTransfer.hpp
#        sublimationTemperature.hpp
        sublimationModel.hpp
        completeSublimation.hpp
        oneDimensionHeatTransfer.hpp
        )

