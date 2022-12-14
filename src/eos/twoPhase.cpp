#include "twoPhase.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"

static inline PetscReal SimpleGasGasDecode(PetscInt dim, const PetscReal *in, PetscReal *out){
    // decode using known alpha
    PetscReal e, rho, gamma1, gamma2, cv1, cv2, R1, R2
    PetscReal alpha = 1, alpha1rho1, Y1, Y2;
    PetscReal T,p,rho1,rho2,e1,e2;
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
    return [rho1, rho2, e1, e2, p, T];
}

static inline PetscReal SimpleGasStiffDecode(PetscInt dim, const PetscReal *in, PetscReal *out){
    // decode using known alpha
    PetscReal e, rho, gamma1, gamma2, R1, p02, cv1, cp2;
    PetscReal alpha = 1, alpha1rho1, Y1, Y2;
    PetscReal T,p,rho1,rho2,e1,e2;
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
        e2 = (e2-p02/rho2)*gamma2/cp2*cv1;
        p = (gamma1-1)*rho1* e1;
        T = e1/cv1;
    }
    return [rho1, rho2, e1, e2, p, T];
}

static inline PetscReal SimpleStiffStiffDecode(PetscInt dim, const PetscReal *in, PetscReal *out){
    // decode using known alpha
    PetscReal e, rho, gamma1, gamma2, p01, p02, cp1, cp2;
    PetscReal alpha = 1, alpha1rho1, Y1, Y2;
    PetscReal T,p,rho1,rho2,e1,e2;
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
        e2 = (e2-p02/rho2)*gamma2/cp2*cp1/gamma1 + p01/rho1;
        p = (gamma1-1)*rho1* e1 - gamma1*p01;
        T = gamma1/cp1*(e1 - p01/rho1);
    }
    return [rho1, rho2, e1, e2, p, T];
}

ablate::eos::TwoPhase::TwoPhase(std::shared_ptr<eos::EOS> eos1, std::shared_ptr<eos::EOS> eos2)
    : eos(std::move(eos1)), eos(std::move(eos2)) {
    // set default values for options
    // need to call eos1 eos2 to fill parameters structure
}

void ablate::eos::TwoPhase::View(std::ostream &stream) const {
    stream << "EOS1: " << type << std::endl;
}

ablate::eos::ThermodynamicFunction ablate::eos::TwoPhase::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) {return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TwoPhase requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicFunction{.function = thermodynamicFunctions.at(property).first,
                                 .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents -2, .eulerOffset = eulerField->offset, .parameters = parameters})};
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::TwoPhase::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) {return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TwoPhase requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicTemperatureFunction{
        .function = thermodynamicFunctions.at(property).second,
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .parameters = parameters})
    };
}

ablate::eos::FieldFunction  ablate::eos::TwoPhase::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2) const {
    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
                auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density, use EOS1 or one with larger volumeFraction?
                // call eos here

                // compute the sensible internal energy

                // convert to total sensibleEnergy

            };
            }
            // other if statements here
    }
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
    *p = 2; // replace with actual value from simple decode
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::PressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    // calculate pressure using temperature from simple decode
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::TemperatureFunction(const PetscReal *conserved, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    // calculate temperature from simple decode
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

    // Compute pressure
    PetscReal p = 2; // replace with simple decode

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

    // Compute pressure
    PetscReal p = 2; // replace with simple decode

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeFunction(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    // call EOS1 EOS2 for cv1 cv2
    // simple decode for Y1 Y2 T rho1 rho2
//    PetscReal w1 =
//    (*specificHeat) = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T
    //    ;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // same as specificHeatConstantVolumeFunction
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureFunction(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    // dont know how to refer to alpha1rho1 field, below conserved[0] = alpha1rho1
    PetscReal Y1 = conserved[0] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[0]) / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    (*specificHeat) = Y1 * parameters.Cp1 + Y2 * parameters.Cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // same as specificHeatConstantPressureFunction
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    // dont know how to refer to alpha1rho1 field, below conserved[0] = alpha1rho1
    PetscReal Y1 = conserved[0] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[0]) / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    (*specificHeat) = Y1 * parameters.Cp1 + Y2 * parameters.Cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundFunction(const PetscReal *conserved, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    // *a = PetscSqrtReal(at_mix^2 + Gamma^2 * cv_mix * T)
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // same as speedOfSoundFunction, isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    // *a = PetscSqrtReal(at_mix^2 + Gamma^2 * cv_mix * T)
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
