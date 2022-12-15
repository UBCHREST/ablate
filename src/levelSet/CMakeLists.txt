target_sources(ablateLibrary
        PRIVATE
        #lsSupport.cpp
        LS-VOF.cpp
        levelSetFields.cpp
        levelSetSolver.cpp

        PUBLIC
        #lsSupport.hpp
        LS-VOF.hpp
        levelSetFields.hpp
        levelSetSolver.hpp
        )
