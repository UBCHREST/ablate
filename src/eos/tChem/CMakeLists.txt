target_sources(ablateLibrary
        PRIVATE
        temperature.cpp
        sensibleInternalEnergy.cpp
        pressure.cpp
        sensibleEnthalpy.cpp
        speedOfSound.cpp

        PUBLIC
        temperature.hpp
        sensibleInternalEnergy.hpp
        sensibleInternalEnergyFcn.hpp
        pressure.hpp
        sensibleEnthalpy.hpp
        sensibleEnthalpyFcn.hpp
        speedOfSound.hpp
        ignitionZeroDTemperatureThreshold.hpp
        )