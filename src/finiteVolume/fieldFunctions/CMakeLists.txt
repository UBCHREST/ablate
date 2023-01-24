target_sources(ablateLibrary
        PRIVATE
        compressibleFlowState.cpp
        euler.cpp
        massFractions.cpp
        densityMassFractions.cpp
        densityExtraVariables.cpp
        densityProgressVariables.cpp

        PUBLIC
        compressibleFlowState.hpp
        euler.hpp
        massFractions.hpp
        densityMassFractions.hpp
        densityExtraVariables.hpp
        densityProgressVariables.hpp
        )