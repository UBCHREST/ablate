#include "twoPhase.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"

// equation of state for equilibrium mixture of two imiscible fluids with perfect gas or stiffened gas equations of state
// reference Menikoff (2007) "Empirical Equations of States for Solids", book chapter in ShockWave Science and Technology Reference Library: Volume 2
static inline PetscReal SimpleGasGasDecode(PetscInt dim, const PetscReal *in, PetscReal *out){
    // decode using known alpha
    PetscReal alpha = in[0];
    PetscReal alpha1rho1 = in[1];
    PetscReal rho = in[2];
    PetscReal e = in[3];
    PetscReal gamma1 = in[4];
    PetscReal gamma2 = in[5];
    PetscReal R1 = in[6];
    PetscReal R2 = in[7];
    PetscReal cv1 = R1 / (gamma1 - 1);
    PetscReal cv2 = R2 / (gamma2 - 1);
    PetscReal Y1 = alpha1rho1 / rho;
    PetscReal Y2 = (rho - alpha1rho1) / rho;
    PetscReal T,p,rho1,rho2,e1,e2; // initiate output variables
    if (alpha < 10E-16){ // all water
        rho2 = rho;
        e2 = e;
        p = (gamma2 - 1) * rho2 * e2;
        T = e2 * cv2;
        rho1 = p/R1/T;
        e1 = cv1* T;
    } else if (alpha > 1-10E-16){ // all air
        rho1 = rho;
        e1 = e;
        p = (gamma1-1)*rho1* e1;
        T = e1*cv1;
        rho2 = p/R2/T;
        e2 = cv2*T;
    } else{
        rho1 = alpha1rho1/alpha;
        rho2 = (rho - alpha1rho1) / (1-alpha);
        e1 = e / (cv2/cv1*Y2 + Y1);
        e2 = e1 *cv2/cv1;
        p = (gamma1-1)*rho1* e1;
        T = e1/cv1;
    }
    out[0] = rho1;
    out[1] = rho2;
    out[2] = e1;
    out[3] = e2;
    out[4] = p;
    out[5] = T;
    return *out;
}

static inline PetscReal SimpleGasStiffDecode(PetscInt dim, const PetscReal *in, PetscReal *out){
    // decode using known alpha
    PetscReal alpha = in[0];
    PetscReal alpha1rho1 = in[1];
    PetscReal rho = in[2];
    PetscReal e = in[3];
    PetscReal gamma1 = in[4];
    PetscReal gamma2 = in[5];
    PetscReal R1 = in[6];
    PetscReal cp2 = in[7];
    PetscReal p02 = in[8];
    PetscReal cv1 = R1 / (gamma1 - 1);
    PetscReal Y1 = alpha1rho1 / rho;
    PetscReal Y2 = (rho - alpha1rho1) / rho;
    PetscReal T,p,rho1,rho2,e1,e2; // initiate output variables
    if (alpha < 10E-16){ // all water
        rho2 = rho;
        e2 = e;
        p = (gamma2 - 1) * rho2 * e2 - gamma2 * p02;
        T = gamma2/cp2*(e2 - p02/rho2);
        rho1 = p/R1/T;
        e1 = cv1* T;
    } else if (alpha > 1-10E-16){ // all air
        rho1 = rho;
        e1 = e;
        p = (gamma1-1)*rho1* e1;
        T = e1*cv1;
        rho2 = (p + p02)*gamma2/(gamma2-1)/ T/ cp2;
        e2 = cp2/gamma2*T + p02/rho2;
    } else{
        rho1 = alpha1rho1/alpha;
        rho2 = (rho - alpha1rho1) / (1-alpha);
        e2 = (e + Y1*cv1/cp2*gamma2*p02/rho2) / (Y1*cv1/cp2*gamma2+Y2);
        e1 = (e2-p02/rho2)*gamma2/cp2*cv1; // make sure order of solver does not affect answer
        p = (gamma1-1)*rho1* e1;
        T = e1/cv1;
    }
    out[0] = rho1;
    out[1] = rho2;
    out[2] = e1;
    out[3] = e2;
    out[4] = p;
    out[5] = T;
    return *out;
}

static inline PetscReal SimpleStiffStiffDecode(PetscInt dim, const PetscReal *in, PetscReal *out){
    // decode using known alpha
    PetscReal alpha = in[0];
    PetscReal alpha1rho1 = in[1];
    PetscReal rho = in[2];
    PetscReal e = in[3];
    PetscReal gamma1 = in[4];
    PetscReal gamma2 = in[5];
    PetscReal cp1 = in[6];
    PetscReal cp2 = in[7];
    PetscReal p01 = in[8];
    PetscReal p02 = in[9];
    PetscReal Y1 = alpha1rho1 / rho;
    PetscReal Y2 = (rho - alpha1rho1) / rho;
    PetscReal T, p, rho1, rho2, e1, e2; // initiate output variables
    if (alpha < 10E-16){ // all water
        rho2 = rho;
        e2 = e;
        p = (gamma2 - 1) * rho2 * e2 - gamma2 * p02;
        T = gamma2/cp2*(e2 - p02/rho2);
        rho1 = (p+p01)*gamma1/(gamma1-1)/T/cp1;
        e1 = cp1/gamma1*T + p01/rho1;
    } else if (alpha > 1-10E-16){ // all air
        rho1 = rho;
        e1 = e;
        p = (gamma1-1)*rho1* e1 - gamma1*p01;
        T = gamma1/cp1*(e1-p01/rho1);
        rho2 = (p + p02)*gamma2/(gamma2-1)/ T/ cp2;
        e2 = cp2/gamma2*T + p02/rho2;
    } else{
        rho1 = alpha1rho1/alpha;
        rho2 = (rho - alpha1rho1) / (1-alpha);
        e2 = (e + Y1*p02/rho2*gamma2/cp2*cp1/gamma1 - Y1*p01/rho1) / (Y1*gamma2/cp2*cp1/gamma1+Y2);
        e1 = (e2-p02/rho2)*gamma2/cp2*cp1/gamma1 + p01/rho1; // make sure order of solve does not affect answer
        p = (gamma1-1)*rho1* e1 - gamma1*p01;
        T = gamma1/cp1*(e1 - p01/rho1);
    }
    out[0] = rho1;
    out[1] = rho2;
    out[2] = e1;
    out[3] = e2;
    out[4] = p;
    out[5] = T;
    return *out;
}

ablate::eos::TwoPhase::TwoPhase(std::shared_ptr<eos::EOS> eos1, std::shared_ptr<eos::EOS> eos2, std::vector<std::string> species)
    : EOS("twoPhase"), eos1(std::move(eos1)), eos2(std::move(eos2)), species(species) {
    // set parameter values
    // check if both perfect gases, use analytical solution
    auto perfectGasEos1 = std::dynamic_pointer_cast<eos::PerfectGas>(eos1);
    auto perfectGasEos2 = std::dynamic_pointer_cast<eos::PerfectGas>(eos2);
    // check if stiffened gas
    auto stiffenedGasEos1 = std::dynamic_pointer_cast<eos::StiffenedGas>(eos1);
    auto stiffenedGasEos2 = std::dynamic_pointer_cast<eos::StiffenedGas>(eos2);
    if (perfectGasEos1 && perfectGasEos2) {
        parameters.gamma1 = perfectGasEos1->GetSpecificHeatRatio();
        parameters.gamma2 = perfectGasEos2->GetSpecificHeatRatio();
        parameters.rGas1 = perfectGasEos1->GetGasConstant();
        parameters.rGas2 = perfectGasEos2->GetGasConstant();
        parameters.p01 = 0;
        parameters.p02 = 0;
    } else if (perfectGasEos1 && stiffenedGasEos2) {
        parameters.gamma1 = perfectGasEos1->GetSpecificHeatRatio();
        parameters.rGas1 = perfectGasEos1->GetGasConstant();
        parameters.p01 = 0;
        parameters.gamma2 = stiffenedGasEos2->GetSpecificHeatRatio();
        parameters.Cp2 = stiffenedGasEos2->GetSpecificHeatCp();
        parameters.p02 = stiffenedGasEos2->GetReferencePressure();
    } else if (stiffenedGasEos1 && stiffenedGasEos2) {
        parameters.gamma1 = stiffenedGasEos1->GetSpecificHeatRatio();
        parameters.Cp1 = stiffenedGasEos1->GetSpecificHeatCp();
        parameters.p01 = stiffenedGasEos1->GetReferencePressure();
        parameters.gamma2 = stiffenedGasEos2->GetSpecificHeatRatio();
        parameters.Cp2 = stiffenedGasEos2->GetSpecificHeatCp();
        parameters.p02 = stiffenedGasEos2->GetReferencePressure();
    }
//    parameters.numberSpecies = 0;  // not used here, need to add support for species eventually
}

void ablate::eos::TwoPhase::View(std::ostream &stream) const {
    stream << "EOS1: " << type << std::endl;
    // add here if we want to stream all parameters
}

ablate::eos::ThermodynamicFunction ablate::eos::TwoPhase::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) {return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    auto densityVFField = std::find_if(fields.begin(), fields.end(), [](const auto &field) {return field.name == "densityVF";});
    auto volumeFractionField = std::find_if(fields.begin(), fields.end(), [](const auto &field) {return field.name == "volumeFraction";});
    // maybe need to throw error for not having densityVF or volumeFraction fields
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TwoPhase requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicFunction{.function = thermodynamicFunctions.at(property).first,
                                 .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents -2, .eulerOffset = eulerField->offset, .densityVFOffset = densityVFField->offset, .volumeFractionOffset = volumeFractionField->offset, .parameters = parameters})};
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::TwoPhase::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) {return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    auto densityVFField = std::find_if(fields.begin(), fields.end(), [](const auto &field) {return field.name == "densityVF";});
    auto volumeFractionField = std::find_if(fields.begin(), fields.end(), [](const auto &field) {return field.name == "volumeFraction";});
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TwoPhase requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicTemperatureFunction{
        .function = thermodynamicFunctions.at(property).second,
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .densityVFOffset = densityVFField->offset, .volumeFractionOffset = volumeFractionField->offset, .parameters = parameters})
    };
}

ablate::eos::FieldFunction  ablate::eos::TwoPhase::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2) const {
//    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {
//        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
//            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
//                auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
//                // Compute the density
//                //  ***** cannot back out density from pressure and temperature only for two fluids, need alpha *****
//
//                // compute the sensible internal energy
//
//                // convert to total sensibleEnergy
//
//            };
//            }
//            // other if statements here
//    }
    return 0;
}

PetscErrorCode ablate::eos::TwoPhase::PressureFunction(const PetscReal *conserved, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    PetscReal derived[6];
    PetscReal in[10];
    in[0] = conserved[functionContext->volumeFractionOffset];
    in[1] = conserved[functionContext->densityVFOffset];
    in[2] = density;
    in[3] = internalEnergy;
    in[4] = functionContext->parameters.gamma1;
    in[5] = functionContext->parameters.gamma2;
    if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 == 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.rGas2;
        SimpleGasGasDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p02;
        SimpleGasStiffDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 != 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.Cp1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p01;
        in[9] = functionContext->parameters.p02;
        SimpleStiffStiffDecode(functionContext->dim, in, derived);
    }
    *p = derived[4]; // [rho1, rho2, e1, e2, p, T]
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::PressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    PetscReal derived[6];
    PetscReal in[10];
    in[0] = conserved[functionContext->volumeFractionOffset];
    in[1] = conserved[functionContext->densityVFOffset];
    in[2] = density;
    in[3] = internalEnergy;
    in[4] = functionContext->parameters.gamma1;
    in[5] = functionContext->parameters.gamma2;
    if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 == 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.rGas2;
        SimpleGasGasDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p02;
        SimpleGasStiffDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 != 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.Cp1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p01;
        in[9] = functionContext->parameters.p02;
        SimpleStiffStiffDecode(functionContext->dim, in, derived);
    }
    *p = derived[4]; // [rho1, rho2, e1, e2, p, T]
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::TemperatureFunction(const PetscReal *conserved, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    PetscReal derived[6];
    PetscReal in[10];
    in[0] = conserved[functionContext->volumeFractionOffset];
    in[1] = conserved[functionContext->densityVFOffset];
    in[2] = density;
    in[3] = internalEnergy;
    in[4] = functionContext->parameters.gamma1;
    in[5] = functionContext->parameters.gamma2;
    if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 == 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.rGas2;
        SimpleGasGasDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p02;
        SimpleGasStiffDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 != 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.Cp1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p01;
        in[9] = functionContext->parameters.p02;
        SimpleStiffStiffDecode(functionContext->dim, in, derived);
    }
    *temperature = derived[5]; // [rho1, rho2, e1, e2, p, T]
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::TemperatureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return TemperatureFunction(conserved, property, ctx);
}
PetscErrorCode ablate::eos::TwoPhase::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *internalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    *internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::InternalSensibleEnergyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *internalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    // same as internalSensibleEnergyFunction
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    *internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::SensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    PetscReal derived[6];
    PetscReal in[10];
    in[0] = conserved[functionContext->volumeFractionOffset];
    in[1] = conserved[functionContext->densityVFOffset];
    in[2] = density;
    in[3] = internalEnergy;
    in[4] = functionContext->parameters.gamma1;
    in[5] = functionContext->parameters.gamma2;
    if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 == 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.rGas2;
        SimpleGasGasDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p02;
        SimpleGasStiffDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 != 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.Cp1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p01;
        in[9] = functionContext->parameters.p02;
        SimpleStiffStiffDecode(functionContext->dim, in, derived);
    }
    PetscReal p = derived[4]; // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::SensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // same as sensibleEnthalpyFunction, Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    PetscReal derived[6];
    PetscReal in[10];
    in[0] = conserved[functionContext->volumeFractionOffset];
    in[1] = conserved[functionContext->densityVFOffset];
    in[2] = density;
    in[3] = internalEnergy;
    in[4] = functionContext->parameters.gamma1;
    in[5] = functionContext->parameters.gamma2;
    if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 == 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.rGas2;
        SimpleGasGasDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p02;
        SimpleGasStiffDecode(functionContext->dim, in, derived);
    } else if (functionContext->parameters.p01 != 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.Cp1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p01;
        in[9] = functionContext->parameters.p02;
        SimpleStiffStiffDecode(functionContext->dim, in, derived);
    }
    PetscReal p = derived[4]; // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeFunction(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    PetscReal derived[6];
    PetscReal in[10];
    in[0] = conserved[functionContext->volumeFractionOffset];
    in[1] = conserved[functionContext->densityVFOffset];
    in[2] = density;
    in[3] = internalEnergy;
    in[4] = functionContext->parameters.gamma1;
    in[5] = functionContext->parameters.gamma2;
    PetscReal cv1, cv2, at1, at2; // initialize variables for at_mix
    if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 == 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.rGas2;
        SimpleGasGasDecode(functionContext->dim, in, derived);
        // isothermal sound speeds
        cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
        cv2 = functionContext->parameters.rGas2 / (functionContext->parameters.gamma2 - 1);
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * derived[5]); // ideal gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) * cv2 * derived[5]); // ideal gas eos
    } else if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p02;
        SimpleGasStiffDecode(functionContext->dim, in, derived);
        cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
        cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * derived[5]); // ideal gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1)/functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * derived[5]); // stiffened gas eos
    } else if (functionContext->parameters.p01 != 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.Cp1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p01;
        in[9] = functionContext->parameters.p02;
        SimpleStiffStiffDecode(functionContext->dim, in, derived);
        cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
        cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1)/functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * derived[5]); // stiffened gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1)/functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * derived[5]); // stiffened gas eos
    } else{
        throw std::invalid_argument("TwoPhase::SpecificHeatConstantVolumeFunction cannot calculate Cv_mix for other EOS, must be perfect/stiffened gas combination.");
    }
    PetscReal rho1 = derived[0];
    PetscReal rho2 = derived[1];
    PetscReal T = derived[5]; // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    (*specificHeat) = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix, same as specificHeatConstantVolumeFunction
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    PetscReal derived[6];
    PetscReal in[10];
    in[0] = conserved[functionContext->volumeFractionOffset];
    in[1] = conserved[functionContext->densityVFOffset];
    in[2] = density;
    in[3] = internalEnergy;
    in[4] = functionContext->parameters.gamma1;
    in[5] = functionContext->parameters.gamma2;
    PetscReal cv1, cv2, at1, at2; // initialize variables for at_mix
    if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 == 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.rGas2;
        SimpleGasGasDecode(functionContext->dim, in, derived);
        // isothermal sound speeds
        cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
        cv2 = functionContext->parameters.rGas2 / (functionContext->parameters.gamma2 - 1);
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * derived[5]); // ideal gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) * cv2 * derived[5]); // ideal gas eos
    } else if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p02;
        SimpleGasStiffDecode(functionContext->dim, in, derived);
        cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
        cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * derived[5]); // ideal gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1)/functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * derived[5]); // stiffened gas eos
    } else if (functionContext->parameters.p01 != 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.Cp1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p01;
        in[9] = functionContext->parameters.p02;
        SimpleStiffStiffDecode(functionContext->dim, in, derived);
        cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
        cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1)/functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * derived[5]); // stiffened gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1)/functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * derived[5]); // stiffened gas eos
    } else{
        throw std::invalid_argument("TwoPhase::SpecificHeatConstantVolumeFunction cannot calculate Cv_mix for other EOS, must be perfect/stiffened gas combination.");
    }
    PetscReal rho1 = derived[0];
    PetscReal rho2 = derived[1]; // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    (*specificHeat) = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureFunction(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal Y1 = conserved[0] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[functionContext->densityVFOffset]) / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    // mixed specific heat constant pressure
    (*specificHeat) = Y1 * parameters.Cp1 + Y2 * parameters.Cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // same as specificHeatConstantPressureFunction
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal Y1 = conserved[0] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[functionContext->densityVFOffset]) / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    // mixed specific heat constant pressure
    (*specificHeat) = Y1 * parameters.Cp1 + Y2 * parameters.Cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundFunction(const PetscReal *conserved, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    PetscReal derived[6];
    PetscReal in[10];
    in[0] = conserved[functionContext->volumeFractionOffset];
    in[1] = conserved[functionContext->densityVFOffset];
    in[2] = density;
    in[3] = internalEnergy;
    in[4] = functionContext->parameters.gamma1;
    in[5] = functionContext->parameters.gamma2;
    PetscReal cv1, cv2, at1, at2; // initialize variables for at_mix
    if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 == 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.rGas2;
        SimpleGasGasDecode(functionContext->dim, in, derived);
        // isothermal sound speeds
        cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
        cv2 = functionContext->parameters.rGas2 / (functionContext->parameters.gamma2 - 1);
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * derived[5]); // ideal gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) * cv2 * derived[5]); // ideal gas eos
    } else if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p02;
        SimpleGasStiffDecode(functionContext->dim, in, derived);
        cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
        cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * derived[5]); // ideal gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1)/functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * derived[5]); // stiffened gas eos
    } else if (functionContext->parameters.p01 != 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.Cp1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p01;
        in[9] = functionContext->parameters.p02;
        SimpleStiffStiffDecode(functionContext->dim, in, derived);
        cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
        cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1)/functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * derived[5]); // stiffened gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1)/functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * derived[5]); // stiffened gas eos
    } else{
        throw std::invalid_argument("TwoPhase::SpecificHeatConstantVolumeFunction cannot calculate Cv_mix for other EOS, must be perfect/stiffened gas combination.");
    }
    PetscReal rho1 = derived[0];
    PetscReal rho2 = derived[1];
    PetscReal T = derived[5]; // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    PetscReal cv_mix = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    // mixed isothermal sound speed
    PetscReal at_mix = PetscSqrtReal(1 / (w1 + w2));
    PetscReal Gamma = (w1*cv1*(gamma1-1)*rho1 + w2*cv2*(gamma2-1)*rho2) / ((w1+w2)*cv_mix*T*density);
    // mixed isentropic sound speed
    *a = PetscSqrtReal(PetscSqr(at_mix) + PetscSqr(Gamma) * cv_mix * T);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // same as speedOfSoundFunction, isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++){
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    PetscReal derived[6];
    PetscReal in[10];
    in[0] = conserved[functionContext->volumeFractionOffset];
    in[1] = conserved[functionContext->densityVFOffset];
    in[2] = density;
    in[3] = internalEnergy;
    in[4] = functionContext->parameters.gamma1;
    in[5] = functionContext->parameters.gamma2;
    PetscReal cv1, cv2, at1, at2; // initialize variables for at_mix
    if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 == 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.rGas2;
        SimpleGasGasDecode(functionContext->dim, in, derived);
        // isothermal sound speeds
        cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
        cv2 = functionContext->parameters.rGas2 / (functionContext->parameters.gamma2 - 1);
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * derived[5]); // ideal gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) * cv2 * derived[5]); // ideal gas eos
    } else if (functionContext->parameters.p01 == 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.rGas1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p02;
        SimpleGasStiffDecode(functionContext->dim, in, derived);
        cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
        cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * derived[5]); // ideal gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1)/functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * derived[5]); // stiffened gas eos
    } else if (functionContext->parameters.p01 != 0 && functionContext->parameters.p02 != 0){
        in[6] = functionContext->parameters.Cp1;
        in[7] = functionContext->parameters.Cp2;
        in[8] = functionContext->parameters.p01;
        in[9] = functionContext->parameters.p02;
        SimpleStiffStiffDecode(functionContext->dim, in, derived);
        cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
        cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
        at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1)/functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * derived[5]); // stiffened gas eos
        at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1)/functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * derived[5]); // stiffened gas eos
    } else{
        throw std::invalid_argument("TwoPhase::SpecificHeatConstantVolumeFunction cannot calculate Cv_mix for other EOS, must be perfect/stiffened gas combination.");
    }
    PetscReal rho1 = derived[0];
    PetscReal rho2 = derived[1]; // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    PetscReal cv_mix = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    // mixed isothermal sound speed
    PetscReal at_mix = PetscSqrtReal(1 / (w1 + w2));
    PetscReal Gamma = (w1*cv1*(gamma1-1)*rho1 + w2*cv2*(gamma2-1)*rho2) / ((w1+w2)*cv_mix*T*density);
    // mixed isentropic sound speed
    *a = PetscSqrtReal(PetscSqr(at_mix) + PetscSqr(Gamma) * cv_mix * T);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    for (PetscInt s = 0; s < parameters.numberSpecies; s++) {
        hi[s] = 0.0;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return SpeciesSensibleEnthalpyFunction(conserved, property, ctx);
}
PetscErrorCode ablate::eos::TwoPhase::DensityFunction(const PetscReal *conserved, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::DensityTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::TwoPhase, "two phase eos", ARG(ablate::eos::EOS, "eos1", "eos for fluid 1, must be prefect or stiffened gas."),
         ARG(ablate::eos::EOS, "eos2", "eos for fluid 2, must be perfect or stiffened gas."));
