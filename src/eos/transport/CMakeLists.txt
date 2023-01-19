target_sources(ablateLibrary
        PRIVATE
        constant.cpp
        sutherland.cpp
        twoPhaseTransport.cpp

        PUBLIC
        transportModel.hpp
        constant.hpp
        sutherland.hpp
        twoPhaseTransport.hpp
        )