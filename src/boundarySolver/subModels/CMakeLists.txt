target_sources(ablateLibrary
        PRIVATE
#        solidHeatTransfer.cpp
#        solidHeatTransferFactory.cpp
#        sublimationTemperature.cpp
        completeSublimation.cpp

        PUBLIC
#        solidHeatTransfer.hpp
#        sublimationTemperature.hpp
        sublimationModel.hpp
        completeSublimation.hpp
        )

