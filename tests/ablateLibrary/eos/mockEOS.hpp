#ifndef ABLATELIBRARY_MOCKEOS_HPP
#define ABLATELIBRARY_MOCKEOS_HPP

#include <map>
#include <ostream>
#include <string>
#include "eos/eos.hpp"
#include "gmock/gmock.h"

namespace ablateTesting::eos {

class MockEOS : public ablate::eos::EOS {
   public:
    MockEOS() : ablate::eos::EOS("MockEOS") {}

    MOCK_METHOD(void, View, (std::ostream & stream), (override, const));

    MOCK_METHOD(ablate::eos::DecodeStateFunction, GetDecodeStateFunction, (), (override));
    MOCK_METHOD(void*, GetDecodeStateContext, (), (override));
    MOCK_METHOD(ablate::eos::ComputeTemperatureFunction, GetComputeTemperatureFunction, (), (override));
    MOCK_METHOD(void*, GetComputeTemperatureContext, (), (override));
    MOCK_METHOD(ablate::eos::ComputeSpeciesSensibleEnthalpyFunction, GetComputeSpeciesSensibleEnthalpyFunction, (), (override));
    MOCK_METHOD(void*, GetComputeSpeciesSensibleEnthalpyContext, (), (override));
    MOCK_METHOD(ablate::eos::ComputeDensityFunctionFromTemperaturePressure, GetComputeDensityFunctionFromTemperaturePressureFunction, (), (override));
    MOCK_METHOD(void*, GetComputeDensityFunctionFromTemperaturePressureContext, (), (override));
    MOCK_METHOD(ablate::eos::ComputeSensibleInternalEnergyFunction, GetComputeSensibleInternalEnergyFunction, (), (override));
    MOCK_METHOD(void*, GetComputeSensibleInternalEnergyContext, (), (override));
    MOCK_METHOD(ablate::eos::ComputeSensibleEnthalpyFunction, GetComputeSensibleEnthalpyFunction, (), (override));
    MOCK_METHOD(void*, GetComputeSensibleEnthalpyContext, (), (override));
    MOCK_METHOD(ablate::eos::ComputeSpecificHeatFunction, GetComputeSpecificHeatConstantPressureFunction, (), (override));
    MOCK_METHOD(void*, GetComputeSpecificHeatConstantPressureContext, (), (override));
    MOCK_METHOD(ablate::eos::ComputeSpecificHeatFunction, GetComputeSpecificHeatConstantVolumeFunction, (), (override));
    MOCK_METHOD(void*, GetComputeSpecificHeatConstantVolumeContext, (), (override));
    MOCK_METHOD(const std::vector<std::string>&, GetSpecies, (), (const, override));

    MOCK_METHOD(ablate::eos::ThermodynamicFunction, GetThermodynamicFunction, (ablate::eos::ThermodynamicProperty, const std::vector<ablate::domain::Field>&), (const, override));
    MOCK_METHOD(ablate::eos::ThermodynamicTemperatureFunction, GetThermodynamicTemperatureFunction, (ablate::eos::ThermodynamicProperty, const std::vector<ablate::domain::Field>&), (const, override));
    MOCK_METHOD(ablate::eos::FieldFunction, GetFieldFunctionFunction, (const std::string& field, ablate::eos::ThermodynamicProperty, ablate::eos::ThermodynamicProperty), (const, override));
};
}  // namespace ablateTesting::eos

#endif  // ABLATELIBRARY_MOCKEOS_HPP
