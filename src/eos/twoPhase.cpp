#include "twoPhase.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"

#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"

// equation of state for equilibrium mixture of two imiscible fluids with perfect gas or stiffened gas equations of state
// reference Menikoff (2007) "Empirical Equations of States for Solids", book chapter in ShockWave Science and Technology Reference Library: Volume 2
static inline void SimpleGasGasDecode(PetscInt dim, ablate::eos::TwoPhase::DecodeIn *in, ablate::eos::TwoPhase::DecodeOut *out) {
    // decode using known alpha
    PetscReal alpha = in->alpha;
    PetscReal alpha1rho1 = in->alphaRho1;
    PetscReal rho = in->rho;
    PetscReal e = in->e;
    PetscReal gamma1 = in->parameters.gamma1;
    PetscReal gamma2 = in->parameters.gamma2;
    PetscReal R1 = in->parameters.rGas1;
    PetscReal R2 = in->parameters.rGas2;
    PetscReal cv1 = R1 / (gamma1 - 1);
    PetscReal cv2 = R2 / (gamma2 - 1);
    PetscReal Y1 = alpha1rho1 / rho;
    PetscReal Y2 = (rho - alpha1rho1) / rho;
    PetscReal T, p, rho1, rho2, e1, e2;  // initiate output variables
    if (alpha < 10E-16) {                // all water
        rho2 = rho;
        e2 = e;
        p = (gamma2 - 1) * rho2 * e2;
        T = e2 * cv2;
        rho1 = p / R1 / T;
        e1 = cv1 * T;
    } else if (alpha > 1 - 10E-16) {  // all air
        rho1 = rho;
        e1 = e;
        p = (gamma1 - 1) * rho1 * e1;
        T = e1 * cv1;
        rho2 = p / R2 / T;
        e2 = cv2 * T;
    } else {
        rho1 = alpha1rho1 / alpha;
        rho2 = (rho - alpha1rho1) / (1 - alpha);
        e1 = e / (cv2 / cv1 * Y2 + Y1);
        e2 = e1 * cv2 / cv1;
        p = (gamma1 - 1) * rho1 * e1;
        T = e1 / cv1;
    }
    out->rho1 = rho1;
    out->rho2 = rho2;
    out->e1 = e1;
    out->e2 = e2;
    out->p = p;
    out->T = T;
}

static inline void SimpleGasStiffDecode(PetscInt dim, ablate::eos::TwoPhase::DecodeIn *in, ablate::eos::TwoPhase::DecodeOut *out) {
    // decode using known alpha
    PetscReal alpha = in->alpha;
    PetscReal alpha1rho1 = in->alphaRho1;
    PetscReal rho = in->rho;
    PetscReal e = in->e;
    PetscReal gamma1 = in->parameters.gamma1;
    PetscReal gamma2 = in->parameters.gamma2;
    PetscReal R1 = in->parameters.rGas1;
    PetscReal cp2 = in->parameters.Cp2;
    PetscReal p02 = in->parameters.p02;
    PetscReal cv1 = R1 / (gamma1 - 1);
    PetscReal Y1 = alpha1rho1 / rho;
    PetscReal Y2 = (rho - alpha1rho1) / rho;
    PetscReal T, p, rho1, rho2, e1, e2;  // initiate output variables
    if (alpha < 10E-16) {                // all water
        rho2 = rho;
        e2 = e;
        p = (gamma2 - 1) * rho2 * e2 - gamma2 * p02;
        T = gamma2 / cp2 * (e2 - p02 / rho2);
        rho1 = p / R1 / T;
        e1 = cv1 * T;
    } else if (alpha > 1 - 10E-16) {  // all air
        rho1 = rho;
        e1 = e;
        p = (gamma1 - 1) * rho1 * e1;
        T = e1 * cv1;
        rho2 = (p + p02) * gamma2 / (gamma2 - 1) / T / cp2;
        e2 = cp2 / gamma2 * T + p02 / rho2;
    } else {
        rho1 = alpha1rho1 / alpha;
        rho2 = (rho - alpha1rho1) / (1 - alpha);
        e2 = (e + Y1 * cv1 / cp2 * gamma2 * p02 / rho2) / (Y1 * cv1 / cp2 * gamma2 + Y2);
        e1 = (e2 - p02 / rho2) * gamma2 / cp2 * cv1;  // make sure order of solver does not affect answer
        p = (gamma1 - 1) * rho1 * e1;
        T = e1 / cv1;
    }
    out->rho1 = rho1;
    out->rho2 = rho2;
    out->e1 = e1;
    out->e2 = e2;
    out->p = p;
    out->T = T;
}

static inline void SimpleStiffStiffDecode(PetscInt dim, ablate::eos::TwoPhase::DecodeIn *in, ablate::eos::TwoPhase::DecodeOut *out) {
    // decode using known alpha
    PetscReal alpha = in->alpha;
    PetscReal alpha1rho1 = in->alphaRho1;
    PetscReal rho = in->rho;
    PetscReal e = in->e;
    PetscReal gamma1 = in->parameters.gamma1;
    PetscReal gamma2 = in->parameters.gamma2;
    PetscReal cp1 = in->parameters.Cp1;
    PetscReal cp2 = in->parameters.Cp2;
    PetscReal p01 = in->parameters.p01;
    PetscReal p02 = in->parameters.p02;
    PetscReal Y1 = alpha1rho1 / rho;
    PetscReal Y2 = (rho - alpha1rho1) / rho;
    PetscReal T, p, rho1, rho2, e1, e2;  // initiate output variables
    if (alpha < 10E-16) {                // all water
        rho2 = rho;
        e2 = e;
        p = (gamma2 - 1) * rho2 * e2 - gamma2 * p02;
        T = gamma2 / cp2 * (e2 - p02 / rho2);
        rho1 = (p + p01) * gamma1 / (gamma1 - 1) / T / cp1;
        e1 = cp1 / gamma1 * T + p01 / rho1;
    } else if (alpha > 1 - 10E-16) {  // all air
        rho1 = rho;
        e1 = e;
        p = (gamma1 - 1) * rho1 * e1 - gamma1 * p01;
        T = gamma1 / cp1 * (e1 - p01 / rho1);
        rho2 = (p + p02) * gamma2 / (gamma2 - 1) / T / cp2;
        e2 = cp2 / gamma2 * T + p02 / rho2;
    } else {
        rho1 = alpha1rho1 / alpha;
        rho2 = (rho - alpha1rho1) / (1 - alpha);
        e2 = (e + Y1 * p02 / rho2 * gamma2 / cp2 * cp1 / gamma1 - Y1 * p01 / rho1) / (Y1 * gamma2 / cp2 * cp1 / gamma1 + Y2);
        e1 = (e2 - p02 / rho2) * gamma2 / cp2 * cp1 / gamma1 + p01 / rho1;  // make sure order of solve does not affect answer
        p = (gamma1 - 1) * rho1 * e1 - gamma1 * p01;
        T = gamma1 / cp1 * (e1 - p01 / rho1);
    }
    out->rho1 = rho1;
    out->rho2 = rho2;
    out->e1 = e1;
    out->e2 = e2;
    out->p = p;
    out->T = T;
}

ablate::eos::TwoPhase::TwoPhase(std::shared_ptr<eos::EOS> eos1, std::shared_ptr<eos::EOS> eos2) : EOS("twoPhase"), eos1(std::move(eos1)), eos2(std::move(eos2)) {
    // set parameter values
    if (this->eos1 && this->eos2) {
        // check if both perfect gases, use analytical solution
        auto perfectGasEos1 = std::dynamic_pointer_cast<eos::PerfectGas>(this->eos1);
        auto perfectGasEos2 = std::dynamic_pointer_cast<eos::PerfectGas>(this->eos2);
        // check if stiffened gas
        auto stiffenedGasEos1 = std::dynamic_pointer_cast<eos::StiffenedGas>(this->eos1);
        auto stiffenedGasEos2 = std::dynamic_pointer_cast<eos::StiffenedGas>(this->eos2);
        if (perfectGasEos1 && perfectGasEos2) {
            parameters.gamma1 = perfectGasEos1->GetSpecificHeatRatio();
            parameters.gamma2 = perfectGasEos2->GetSpecificHeatRatio();
            parameters.rGas1 = perfectGasEos1->GetGasConstant();
            parameters.rGas2 = perfectGasEos2->GetGasConstant();
            parameters.p01 = 0;
            parameters.p02 = 0;
            parameters.numberSpecies1 = perfectGasEos1->GetSpeciesVariables().size();
            parameters.species1 = perfectGasEos1->GetSpeciesVariables();
            parameters.numberSpecies2 = perfectGasEos2->GetSpeciesVariables().size();
            parameters.species2 = perfectGasEos2->GetSpeciesVariables();
        } else if (perfectGasEos1 && stiffenedGasEos2) {
            parameters.gamma1 = perfectGasEos1->GetSpecificHeatRatio();
            parameters.rGas1 = perfectGasEos1->GetGasConstant();
            parameters.p01 = 0;
            parameters.gamma2 = stiffenedGasEos2->GetSpecificHeatRatio();
            parameters.Cp2 = stiffenedGasEos2->GetSpecificHeatCp();
            parameters.p02 = stiffenedGasEos2->GetReferencePressure();
            parameters.numberSpecies1 = perfectGasEos1->GetSpeciesVariables().size();
            parameters.species1 = perfectGasEos1->GetSpeciesVariables();
            parameters.numberSpecies2 = stiffenedGasEos2->GetSpeciesVariables().size();
            parameters.species2 = stiffenedGasEos2->GetSpeciesVariables();
        } else if (stiffenedGasEos1 && stiffenedGasEos2) {
            parameters.gamma1 = stiffenedGasEos1->GetSpecificHeatRatio();
            parameters.Cp1 = stiffenedGasEos1->GetSpecificHeatCp();
            parameters.p01 = stiffenedGasEos1->GetReferencePressure();
            parameters.gamma2 = stiffenedGasEos2->GetSpecificHeatRatio();
            parameters.Cp2 = stiffenedGasEos2->GetSpecificHeatCp();
            parameters.p02 = stiffenedGasEos2->GetReferencePressure();
            parameters.numberSpecies1 = stiffenedGasEos1->GetSpeciesVariables().size();
            parameters.species1 = stiffenedGasEos1->GetSpeciesVariables();
            parameters.numberSpecies2 = stiffenedGasEos2->GetSpeciesVariables().size();
            parameters.species2 = stiffenedGasEos2->GetSpeciesVariables();
        }
        species.resize(parameters.numberSpecies1 + parameters.numberSpecies2);
        for (PetscInt c = 0; c < parameters.numberSpecies1; c++) {
            species[c] = parameters.species1[c];
        }
        for (PetscInt c = parameters.numberSpecies1; c < parameters.numberSpecies1 + parameters.numberSpecies2; c++) {
            species[c] = parameters.species2[c - parameters.numberSpecies1];
        }
    } else {
        // defaults to air (perfect) and water (stiffened) with no species
        parameters.gamma1 = 1.4;
        parameters.rGas1 = 287.0;  // Cp1 not populated
        parameters.p01 = 0;
        parameters.gamma2 = 1.932;  // rGas2 not populated
        parameters.Cp2 = 8095.08;
        parameters.p02 = 1.1645e9;
        parameters.numberSpecies1 = 0;
        parameters.species1 = {};
        parameters.numberSpecies2 = 0;
        parameters.species2 = {};
        species = {};
    }
}

void ablate::eos::TwoPhase::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    if (eos1 && eos2) {
        stream << *eos1;
        stream << *eos2;
    }
}

ablate::eos::ThermodynamicFunction ablate::eos::TwoPhase::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    auto densityVFField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD; });
    auto volumeFractionField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD; });
    // maybe need to throw error for not having densityVF or volumeFraction fields
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TwoPhase requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    if (parameters.p01 == 0 && parameters.p02 == 0) {  // GasGas case
        return ThermodynamicFunction{.function = thermodynamicFunctionsGasGas.at(property).first,
                                     .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                                                                  .eulerOffset = eulerField->offset,
                                                                                                  .densityVFOffset = densityVFField->offset,
                                                                                                  .volumeFractionOffset = volumeFractionField->offset,
                                                                                                  .parameters = parameters})};
    } else if (parameters.p01 == 0 && parameters.p02 != 0) {  // GasLiquid case
        return ThermodynamicFunction{.function = thermodynamicFunctionsGasLiquid.at(property).first,
                                     .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                                                                  .eulerOffset = eulerField->offset,
                                                                                                  .densityVFOffset = densityVFField->offset,
                                                                                                  .volumeFractionOffset = volumeFractionField->offset,
                                                                                                  .parameters = parameters})};
    } else if (parameters.p01 != 0 && parameters.p02 != 0) {  // LiquidLiquid case
        return ThermodynamicFunction{.function = thermodynamicFunctionsLiquidLiquid.at(property).first,
                                     .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                                                                  .eulerOffset = eulerField->offset,
                                                                                                  .densityVFOffset = densityVFField->offset,
                                                                                                  .volumeFractionOffset = volumeFractionField->offset,
                                                                                                  .parameters = parameters})};
    } else {  // default ?? here default is GasLiquid air/water
        return ThermodynamicFunction{.function = thermodynamicFunctionsGasLiquid.at(property).first,
                                     .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                                                                  .eulerOffset = eulerField->offset,
                                                                                                  .densityVFOffset = densityVFField->offset,
                                                                                                  .volumeFractionOffset = volumeFractionField->offset,
                                                                                                  .parameters = parameters})};
    }
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::TwoPhase::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    auto densityVFField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD; });
    auto volumeFractionField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TwoPhase requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    if (parameters.p01 == 0 && parameters.p02 == 0) {  // GasGas case
        return ThermodynamicTemperatureFunction{.function = thermodynamicFunctionsGasGas.at(property).second,
                                                .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                                                                             .eulerOffset = eulerField->offset,
                                                                                                             .densityVFOffset = densityVFField->offset,
                                                                                                             .volumeFractionOffset = volumeFractionField->offset,
                                                                                                             .parameters = parameters})};
    } else if (parameters.p01 == 0 && parameters.p02 != 0) {  // GasLiquid case
        return ThermodynamicTemperatureFunction{.function = thermodynamicFunctionsGasLiquid.at(property).second,
                                                .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                                                                             .eulerOffset = eulerField->offset,
                                                                                                             .densityVFOffset = densityVFField->offset,
                                                                                                             .volumeFractionOffset = volumeFractionField->offset,
                                                                                                             .parameters = parameters})};
    } else if (parameters.p01 != 0 && parameters.p02 != 0) {  // LiquidLiquid case
        return ThermodynamicTemperatureFunction{.function = thermodynamicFunctionsLiquidLiquid.at(property).second,
                                                .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                                                                             .eulerOffset = eulerField->offset,
                                                                                                             .densityVFOffset = densityVFField->offset,
                                                                                                             .volumeFractionOffset = volumeFractionField->offset,
                                                                                                             .parameters = parameters})};
    } else {  // default ?? here default is GasLiquid air/water
        return ThermodynamicTemperatureFunction{.function = thermodynamicFunctionsGasLiquid.at(property).second,
                                                .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                                                                             .eulerOffset = eulerField->offset,
                                                                                                             .densityVFOffset = densityVFField->offset,
                                                                                                             .volumeFractionOffset = volumeFractionField->offset,
                                                                                                             .parameters = parameters})};
    }
}

ablate::eos::EOSFunction ablate::eos::TwoPhase::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2,
                                                                         std::vector<std::string> otherProperties) const {
    if (otherProperties != std::vector<std::string>{VF} && otherProperties != std::vector<std::string>{VF, YI}) {  // VF not in otherProperties){
        throw std::invalid_argument("ablate::eos::TwoPhase expects other properties to include VF (volume fraction) as first entry, and optionally, YI (species) as second entry");
    }

    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {
        // temperature & pressure & alpha (** note: need volume fraction in otherProperties to back out conserved variables **)
        // Not: This function is used for initializing fields from P,T,vel instead of conserved variables
        //      -> this would mean need to add option: if (field == densityVF)
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density and internal energy for first fluid
                //  ***** cannot back out density from pressure and temperature only for two fluids, need alpha as input to this *****
                PetscReal alpha = yi[0];
                PetscReal rho1, rho2, e1, e2;
                if (parameters.p01 == 0) {
                    rho1 = pressure / (temperature * parameters.rGas1);
                    PetscReal cv1 = parameters.rGas1 / (parameters.gamma1 - 1.0);
                    e1 = temperature * cv1;
                } else if (parameters.p01 != 0) {
                    rho1 = (pressure + parameters.p01) / (parameters.gamma1 - 1) * parameters.gamma1 / parameters.Cp1 / temperature;
                    e1 = temperature * parameters.Cp1 / parameters.gamma1 + parameters.p01 / rho1;
                } else {
                    throw std::invalid_argument("p01 value not valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }
                // density at internal energy for second fluid
                if (parameters.p02 == 0) {
                    rho2 = pressure / (temperature * parameters.rGas2);
                    PetscReal cv2 = parameters.rGas2 / (parameters.gamma2 - 1.0);
                    e2 = temperature * cv2;
                } else if (parameters.p02 != 0) {
                    rho2 = (pressure + parameters.p02) / (parameters.gamma2 - 1) * parameters.gamma2 / parameters.Cp2 / temperature;
                    e2 = temperature * parameters.Cp2 / parameters.gamma2 + parameters.p02 / rho2;
                } else {
                    throw std::invalid_argument(" p02 value not valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }
                PetscReal density = alpha * rho1 + (1 - alpha) * rho2;
                // compute sensible internal energy
                PetscReal Y1 = alpha * rho1 / density;
                PetscReal Y2 = (density - alpha * rho1) / density;
                PetscReal sensibleInternalEnergy = Y1 * e1 + Y2 * e2;

                // convert to total sensibleEnergy
                PetscReal kineticEnergy = 0;
                for (PetscInt d = 0; d < dim; d++) {
                    kineticEnergy += PetscSqr(velocity[d]);
                }
                kineticEnergy *= 0.5;

                conserved[ablate::finiteVolume::CompressibleFlowFields::RHO] = density;
                conserved[ablate::finiteVolume::CompressibleFlowFields::RHOE] = density * (kineticEnergy + sensibleInternalEnergy);

                for (PetscInt d = 0; d < dim; d++) {
                    conserved[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density * velocity[d];
                }
            };
            if (property1 == ThermodynamicProperty::Temperature) {
                return tp;
            } else {
                return [tp](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    tp(temperature, pressure, dim, velocity, yi, conserved);
                };
            }
        }
        // pressure & energy & alpha
        if ((property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy) ||
            (property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure)) {
            auto iep = [this](PetscReal internalSensibleEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal alpha = yi[0];
                PetscReal density;
                if (parameters.p01 == 0 && parameters.p02 == 0) {
                    density = pressure / internalSensibleEnergy * (alpha / (parameters.gamma1 - 1) + (1 - alpha) / (parameters.gamma2 - 1));
                } else if (parameters.p01 == 0 && parameters.p02 != 0) {
                    density = 1 / internalSensibleEnergy * (alpha * pressure / (parameters.gamma1 - 1) + (pressure + parameters.gamma2 * parameters.p02) * (1 - alpha) / (parameters.gamma2 - 1));
                } else if (parameters.p01 != 0 && parameters.p02 != 0) {
                    density =
                        1 / internalSensibleEnergy *
                        (alpha * (pressure + parameters.gamma1 * parameters.p01) / (parameters.gamma1 - 1) + (pressure + parameters.gamma2 * parameters.p02) * (1 - alpha) / (parameters.gamma2 - 1));
                } else {
                    throw std::invalid_argument("no valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }

                // convert to total sensibleEnergy
                PetscReal kineticEnergy = 0;
                for (PetscInt d = 0; d < dim; d++) {
                    kineticEnergy += PetscSqr(velocity[d]);
                }
                kineticEnergy *= 0.5;

                conserved[ablate::finiteVolume::CompressibleFlowFields::RHO] = density;
                conserved[ablate::finiteVolume::CompressibleFlowFields::RHOE] = density * (kineticEnergy + internalSensibleEnergy);

                for (PetscInt d = 0; d < dim; d++) {
                    conserved[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density * velocity[d];
                }
            };

            if (property1 == ThermodynamicProperty::InternalSensibleEnergy) {
                return iep;
            } else {
                return [iep](PetscReal pressure, PetscReal internalSensibleEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    iep(internalSensibleEnergy, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::TwoPhase.");

    } else if (finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density and internal energy for first fluid
                // Note: cannot back out density from pressure and temperature only for two fluids, need alpha as additional input
                PetscReal alpha = yi[0];
                PetscReal rho1;
                if (parameters.p01 == 0) {
                    rho1 = pressure / (temperature * parameters.rGas1);
                } else if (parameters.p01 != 0) {
                    rho1 = (pressure + parameters.p01) / (parameters.gamma1 - 1) * parameters.gamma1 / parameters.Cp1 / temperature;
                } else {
                    throw std::invalid_argument("p01 value not valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }

                conserved[0] = alpha * rho1;
            };
            if (property1 == ThermodynamicProperty::Temperature) {
                return tp;
            } else {
                return [tp](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    tp(temperature, pressure, dim, velocity, yi, conserved);
                };
            }
        }
        // pressure & energy & alpha
        if ((property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy) ||
            (property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure)) {
            auto iep = [this](PetscReal internalSensibleEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal alpha = yi[0];
                PetscReal density, cv1, cv2, rho1, rhoe;
                if (parameters.p01 == 0 && parameters.p02 == 0) {
                    density = pressure / internalSensibleEnergy * (alpha / (parameters.gamma1 - 1) + (1 - alpha) / (parameters.gamma2 - 1));
                    cv1 = parameters.rGas1 / (parameters.gamma1 - 1.0);
                    cv2 = parameters.rGas2 / (parameters.gamma2 - 1.0);
                    rhoe = pressure / (parameters.gamma1 - 1);
                    rho1 = cv2 / cv1 * density * rhoe / (density * internalSensibleEnergy + alpha * rhoe * (cv2 / cv1 - 1));
                } else if (parameters.p01 == 0 && parameters.p02 != 0) {
                    density = 1 / internalSensibleEnergy * (alpha * pressure / (parameters.gamma1 - 1) + (pressure + parameters.gamma2 * parameters.p02) * (1 - alpha) / (parameters.gamma2 - 1));
                    cv1 = parameters.rGas1 / (parameters.gamma1 - 1.0);
                    rhoe = pressure / (parameters.gamma1 - 1);
                    PetscReal coeffs = parameters.Cp2 / cv1 / parameters.gamma2;
                    rho1 = coeffs * density * rhoe / (density * internalSensibleEnergy + alpha * rhoe * (coeffs - 1) - parameters.p02 * (1 - alpha));
                } else if (parameters.p01 != 0 && parameters.p02 != 0) {
                    density =
                        1 / internalSensibleEnergy *
                        (alpha * (pressure + parameters.gamma1 * parameters.p01) / (parameters.gamma1 - 1) + (pressure + parameters.gamma2 * parameters.p02) * (1 - alpha) / (parameters.gamma2 - 1));
                    rhoe = (pressure + parameters.gamma1 * parameters.p01) / (parameters.gamma1 - 1);
                    PetscReal coeffs = parameters.gamma1 * parameters.Cp2 / parameters.Cp1 / parameters.gamma2;
                    rho1 =
                        coeffs * density * (rhoe - parameters.p01) / (density * internalSensibleEnergy - parameters.p02 * (1 - alpha) - alpha * parameters.p01 * coeffs + alpha * rhoe * (coeffs - 1));
                } else {
                    throw std::invalid_argument("no valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }

                conserved[0] = alpha * rho1;
            };

            if (property1 == ThermodynamicProperty::InternalSensibleEnergy) {
                return iep;
            } else {
                return [iep](PetscReal pressure, PetscReal internalSensibleEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    iep(internalSensibleEnergy, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::TwoPhase.");

    } else if (finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD == field) {
        if (property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) {
            return [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density of fluid 1
                PetscReal rho1, rho2;
                if (parameters.p01 == 0) {
                    rho1 = pressure / (temperature * parameters.rGas1);
                } else if (parameters.p01 != 0) {
                    rho1 = (pressure + parameters.p01) / (parameters.gamma1 - 1) * parameters.gamma1 / parameters.Cp1 / temperature;
                } else {
                    throw std::invalid_argument("p01 value not valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }
                // density for second fluid
                if (parameters.p02 == 0) {
                    rho2 = pressure / (temperature * parameters.rGas2);
                } else if (parameters.p02 != 0) {
                    rho2 = (pressure + parameters.p02) / (parameters.gamma2 - 1) * parameters.gamma2 / parameters.Cp2 / temperature;
                } else {
                    throw std::invalid_argument(" p02 value not valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }

                for (PetscInt c = 0; c < parameters.numberSpecies1; c++) {  // species of fluid 1
                    conserved[c] = rho1 * yi[c + 1];                        // first one is alpha
                }
                for (PetscInt c = parameters.numberSpecies1; c < parameters.numberSpecies1 + parameters.numberSpecies2; c++) {  // species of fluid 2
                    conserved[c] = rho2 * yi[c + 1];                                                                            // first one is alpha
                }
            };
        } else if (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature) {
            return [this](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal rho1, rho2;
                if (parameters.p01 == 0) {
                    rho1 = pressure / (temperature * parameters.rGas1);
                } else if (parameters.p01 != 0) {
                    rho1 = (pressure + parameters.p01) / (parameters.gamma1 - 1) * parameters.gamma1 / parameters.Cp1 / temperature;
                } else {
                    throw std::invalid_argument("p01 value not valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }
                // density for second fluid
                if (parameters.p02 == 0) {
                    rho2 = pressure / (temperature * parameters.rGas2);
                } else if (parameters.p02 != 0) {
                    rho2 = (pressure + parameters.p02) / (parameters.gamma2 - 1) * parameters.gamma2 / parameters.Cp2 / temperature;
                } else {
                    throw std::invalid_argument(" p02 value not valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }

                for (PetscInt c = 0; c < parameters.numberSpecies1; c++) {  // species of fluid 1
                    conserved[c] = rho1 * yi[c + 1];                        // first one is alpha
                }
                for (PetscInt c = parameters.numberSpecies1; c < parameters.numberSpecies1 + parameters.numberSpecies2; c++) {
                    conserved[c] = rho2 * yi[c + 1];  // first one is alpha
                }
            };
        } else if (property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure) {
            return [this](PetscReal internalSensibleEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal alpha = yi[0];
                PetscReal density, cv1, cv2, rho1, rhoe, rho2;
                if (parameters.p01 == 0 && parameters.p02 == 0) {
                    density = pressure / internalSensibleEnergy * (alpha / (parameters.gamma1 - 1) + (1 - alpha) / (parameters.gamma2 - 1));
                    cv1 = parameters.rGas1 / (parameters.gamma1 - 1.0);
                    cv2 = parameters.rGas2 / (parameters.gamma2 - 1.0);
                    rhoe = pressure / (parameters.gamma1 - 1);
                    rho1 = cv2 / cv1 * density * rhoe / (density * internalSensibleEnergy + alpha * rhoe * (cv2 / cv1 - 1));
                    rho2 = pressure / (parameters.gamma2 - 1) / internalSensibleEnergy * (alpha * rho1 / density * cv1 / cv2 + (density - alpha * rho1) / density);
                } else if (parameters.p01 == 0 && parameters.p02 != 0) {
                    density = 1 / internalSensibleEnergy * (alpha * pressure / (parameters.gamma1 - 1) + (pressure + parameters.gamma2 * parameters.p02) * (1 - alpha) / (parameters.gamma2 - 1));
                    cv1 = parameters.rGas1 / (parameters.gamma1 - 1.0);
                    rhoe = pressure / (parameters.gamma1 - 1);
                    PetscReal coeffs = parameters.Cp2 / cv1 / parameters.gamma2;
                    rho1 = coeffs * density * rhoe / (density * internalSensibleEnergy + alpha * rhoe * (coeffs - 1) - parameters.p02 * (1 - alpha));
                    PetscReal denom = alpha * rho1 / density / coeffs + (density - alpha * rho1) / density;
                    rho2 = denom * (pressure + parameters.gamma2 * parameters.p02) / (parameters.gamma2 - 1) / internalSensibleEnergy -
                           alpha * rho1 / density / coeffs * parameters.p02 / internalSensibleEnergy;
                } else if (parameters.p01 != 0 && parameters.p02 != 0) {
                    density =
                        1 / internalSensibleEnergy *
                        (alpha * (pressure + parameters.gamma1 * parameters.p01) / (parameters.gamma1 - 1) + (pressure + parameters.gamma2 * parameters.p02) * (1 - alpha) / (parameters.gamma2 - 1));
                    rhoe = (pressure + parameters.gamma1 * parameters.p01) / (parameters.gamma1 - 1);
                    PetscReal coeffs = parameters.gamma1 * parameters.Cp2 / parameters.Cp1 / parameters.gamma2;
                    rho1 =
                        coeffs * density * (rhoe - parameters.p01) / (density * internalSensibleEnergy - parameters.p02 * (1 - alpha) - alpha * parameters.p01 * coeffs + alpha * rhoe * (coeffs - 1));
                    PetscReal denom = alpha * rho1 / density / coeffs + (density - alpha * rho1) / density;
                    rho2 = 1 / (internalSensibleEnergy - alpha * parameters.p01 / density) *
                           (denom * (pressure + parameters.gamma2 * parameters.p02) / (parameters.gamma2 - 1) - alpha * rho1 / density * parameters.p02 / coeffs);
                } else {
                    throw std::invalid_argument("no valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }

                for (PetscInt c = 0; c < parameters.numberSpecies1; c++) {
                    conserved[c] = rho1 * yi[c + 1];  // first one is alpha
                }
                for (PetscInt c = parameters.numberSpecies1; c < parameters.numberSpecies1 + parameters.numberSpecies2; c++) {
                    conserved[c] = rho2 * yi[c + 1];
                }
            };
        } else if (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy) {
            return [this](PetscReal pressure, PetscReal internalSensibleEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal alpha = yi[0];
                PetscReal density, cv1, cv2, rho1, rhoe, rho2;
                if (parameters.p01 == 0 && parameters.p02 == 0) {
                    density = pressure / internalSensibleEnergy * (alpha / (parameters.gamma1 - 1) + (1 - alpha) / (parameters.gamma2 - 1));
                    cv1 = parameters.rGas1 / (parameters.gamma1 - 1.0);
                    cv2 = parameters.rGas2 / (parameters.gamma2 - 1.0);
                    rhoe = pressure / (parameters.gamma1 - 1);
                    rho1 = cv2 / cv1 * density * rhoe / (density * internalSensibleEnergy + alpha * rhoe * (cv2 / cv1 - 1));
                    rho2 = pressure / (parameters.gamma2 - 1) / internalSensibleEnergy * (alpha * rho1 / density * cv1 / cv2 + (density - alpha * rho1) / density);
                } else if (parameters.p01 == 0 && parameters.p02 != 0) {
                    density = 1 / internalSensibleEnergy * (alpha * pressure / (parameters.gamma1 - 1) + (pressure + parameters.gamma2 * parameters.p02) * (1 - alpha) / (parameters.gamma2 - 1));
                    cv1 = parameters.rGas1 / (parameters.gamma1 - 1.0);
                    rhoe = pressure / (parameters.gamma1 - 1);
                    PetscReal coeffs = parameters.Cp2 / cv1 / parameters.gamma2;
                    rho1 = coeffs * density * rhoe / (density * internalSensibleEnergy + alpha * rhoe * (coeffs - 1) - parameters.p02 * (1 - alpha));
                    PetscReal denom = alpha * rho1 / density / coeffs + (density - alpha * rho1) / density;
                    rho2 = denom * (pressure + parameters.gamma2 * parameters.p02) / (parameters.gamma2 - 1) / internalSensibleEnergy -
                           alpha * rho1 / density / coeffs * parameters.p02 / internalSensibleEnergy;
                } else if (parameters.p01 != 0 && parameters.p02 != 0) {
                    density =
                        1 / internalSensibleEnergy *
                        (alpha * (pressure + parameters.gamma1 * parameters.p01) / (parameters.gamma1 - 1) + (pressure + parameters.gamma2 * parameters.p02) * (1 - alpha) / (parameters.gamma2 - 1));
                    rhoe = (pressure + parameters.gamma1 * parameters.p01) / (parameters.gamma1 - 1);
                    PetscReal coeffs = parameters.gamma1 * parameters.Cp2 / parameters.Cp1 / parameters.gamma2;
                    rho1 =
                        coeffs * density * (rhoe - parameters.p01) / (density * internalSensibleEnergy - parameters.p02 * (1 - alpha) - alpha * parameters.p01 * coeffs + alpha * rhoe * (coeffs - 1));
                    PetscReal denom = alpha * rho1 / density / coeffs + (density - alpha * rho1) / density;
                    rho2 = 1 / (internalSensibleEnergy - alpha * parameters.p01 / density) *
                           (denom * (pressure + parameters.gamma2 * parameters.p02) / (parameters.gamma2 - 1) - alpha * rho1 / density * parameters.p02 / coeffs);
                } else {
                    throw std::invalid_argument("no valid, TwoPhase::GetFieldFunction needs perfect/stiffened gas eos combination");
                }
                for (PetscInt c = 0; c < parameters.numberSpecies1; c++) {
                    conserved[c] = rho1 * yi[c + 1];  // first one is alpha
                }
                for (PetscInt c = parameters.numberSpecies1; c < parameters.numberSpecies1 + parameters.numberSpecies2; c++) {
                    conserved[c] = rho2 * yi[c + 1];
                }
            };
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::TwoPhase.");
    } else {
        throw std::invalid_argument("Unknown field type " + field + " for ablate::eos::TwoPhase.");
    }
}

PetscErrorCode ablate::eos::TwoPhase::PressureFunctionGasGas(const PetscReal *conserved, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasGasDecode(functionContext->dim, &decodeIn, &decodeOut);
    *p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::PressureFunctionGasLiquid(const PetscReal *conserved, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    *p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::PressureFunctionLiquidLiquid(const PetscReal *conserved, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleStiffStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    *p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::PressureTemperatureFunctionGasGas(const PetscReal *conserved, PetscReal T, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasGasDecode(functionContext->dim, &decodeIn, &decodeOut);
    *p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::PressureTemperatureFunctionGasLiquid(const PetscReal *conserved, PetscReal T, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    *p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::PressureTemperatureFunctionLiquidLiquid(const PetscReal *conserved, PetscReal T, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleStiffStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    *p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::TemperatureFunctionGasGas(const PetscReal *conserved, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasGasDecode(functionContext->dim, &decodeIn, &decodeOut);
    *temperature = decodeOut.T;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::TemperatureFunctionGasLiquid(const PetscReal *conserved, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    *temperature = decodeOut.T;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::TemperatureFunctionLiquidLiquid(const PetscReal *conserved, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleStiffStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    *temperature = decodeOut.T;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::TemperatureTemperatureFunctionGasGas(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return TemperatureFunctionGasGas(conserved, property, ctx);
}
PetscErrorCode ablate::eos::TwoPhase::TemperatureTemperatureFunctionGasLiquid(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return TemperatureFunctionGasLiquid(conserved, property, ctx);
}
PetscErrorCode ablate::eos::TwoPhase::TemperatureTemperatureFunctionLiquidLiquid(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return TemperatureFunctionLiquidLiquid(conserved, property, ctx);
}

PetscErrorCode ablate::eos::TwoPhase::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *internalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // get the velocity for kinetic energy
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
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
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    *internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::SensibleEnthalpyFunctionGasGas(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = sensibleInternalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasGasDecode(functionContext->dim, &decodeIn, &decodeOut);
    PetscReal p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SensibleEnthalpyFunctionGasLiquid(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = sensibleInternalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    PetscReal p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SensibleEnthalpyFunctionLiquidLiquid(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = sensibleInternalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleStiffStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    PetscReal p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::SensibleEnthalpyTemperatureFunctionGasGas(const PetscReal *conserved, PetscReal T, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // same as sensibleEnthalpyFunction, Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = sensibleInternalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasGasDecode(functionContext->dim, &decodeIn, &decodeOut);
    PetscReal p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SensibleEnthalpyTemperatureFunctionGasLiquid(const PetscReal *conserved, PetscReal T, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // same as sensibleEnthalpyFunction, Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = sensibleInternalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleGasStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    PetscReal p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SensibleEnthalpyTemperatureFunctionLiquidLiquid(const PetscReal *conserved, PetscReal T, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // same as sensibleEnthalpyFunction, Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = sensibleInternalEnergy;
    decodeIn.parameters = functionContext->parameters;

    SimpleStiffStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    PetscReal p = decodeOut.p;  // [rho1, rho2, e1, e2, p, T]

    // compute enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeFunctionGasGas(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleGasGasDecode(functionContext->dim, &decodeIn, &decodeOut);
    // isothermal sound speeds
    cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
    cv2 = functionContext->parameters.rGas2 / (functionContext->parameters.gamma2 - 1);
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * decodeOut.T);  // ideal gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) * cv2 * decodeOut.T);  // ideal gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
    PetscReal T = decodeOut.T;  // [rho1, rho2, e1, e2, p, T]
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
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeFunctionGasLiquid(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleGasStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * decodeOut.T);                                                                   // ideal gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * decodeOut.T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
    PetscReal T = decodeOut.T;  // [rho1, rho2, e1, e2, p, T]
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
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeFunctionLiquidLiquid(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleStiffStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) / functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * decodeOut.T);  // stiffened gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * decodeOut.T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
    PetscReal T = decodeOut.T;  // [rho1, rho2, e1, e2, p, T]
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
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeTemperatureFunctionGasGas(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix, same as specificHeatConstantVolumeFunction
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleGasGasDecode(functionContext->dim, &decodeIn, &decodeOut);
    // isothermal sound speeds
    cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
    cv2 = functionContext->parameters.rGas2 / (functionContext->parameters.gamma2 - 1);
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * T);  // ideal gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) * cv2 * T);  // ideal gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
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
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeTemperatureFunctionGasLiquid(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix, same as specificHeatConstantVolumeFunction
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleGasStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * T);                                                                   // ideal gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
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
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantVolumeTemperatureFunctionLiquidLiquid(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // cv_mix, same as specificHeatConstantVolumeFunction
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleStiffStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) / functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * T);  // stiffened gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
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

PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureFunctionGasGas(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[functionContext->densityVFOffset]) /
                   conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal cp1, cp2;
    cp1 = parameters.gamma1 * parameters.rGas1 / (parameters.gamma1 - 1);
    cp2 = parameters.gamma2 * parameters.rGas2 / (parameters.gamma2 - 1);

    // mixed specific heat constant pressure
    (*specificHeat) = Y1 * cp1 + Y2 * cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureFunctionGasLiquid(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[functionContext->densityVFOffset]) /
                   conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal cp1, cp2;
    cp1 = parameters.gamma1 * parameters.rGas1 / (parameters.gamma1 - 1);
    cp2 = parameters.Cp2;

    // mixed specific heat constant pressure
    (*specificHeat) = Y1 * cp1 + Y2 * cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureFunctionLiquidLiquid(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[functionContext->densityVFOffset]) /
                   conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal cp1, cp2;
    cp1 = parameters.Cp1;
    cp2 = parameters.Cp2;

    // mixed specific heat constant pressure
    (*specificHeat) = Y1 * cp1 + Y2 * cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureTemperatureFunctionGasGas(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // same as specificHeatConstantPressureFunction
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[functionContext->densityVFOffset]) /
                   conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal cp1, cp2;
    cp1 = parameters.gamma1 * parameters.rGas1 / (parameters.gamma1 - 1);
    cp2 = parameters.gamma2 * parameters.rGas2 / (parameters.gamma2 - 1);

    // mixed specific heat constant pressure
    (*specificHeat) = Y1 * cp1 + Y2 * cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureTemperatureFunctionGasLiquid(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // same as specificHeatConstantPressureFunction
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[functionContext->densityVFOffset]) /
                   conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal cp1, cp2;
    cp1 = parameters.gamma1 * parameters.rGas1 / (parameters.gamma1 - 1);
    cp2 = parameters.Cp2;

    // mixed specific heat constant pressure
    (*specificHeat) = Y1 * cp1 + Y2 * cp2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpecificHeatConstantPressureTemperatureFunctionLiquidLiquid(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    // same as specificHeatConstantPressureFunction
    auto functionContext = (FunctionContext *)ctx;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal Y2 = (conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO] - conserved[functionContext->densityVFOffset]) /
                   conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal cp1, cp2;
    cp1 = parameters.Cp1;
    cp2 = parameters.Cp2;

    // mixed specific heat constant pressure
    (*specificHeat) = Y1 * cp1 + Y2 * cp2;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundFunctionGasGas(const PetscReal *conserved, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleGasGasDecode(functionContext->dim, &decodeIn, &decodeOut);
    // isothermal sound speeds
    cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
    cv2 = functionContext->parameters.rGas2 / (functionContext->parameters.gamma2 - 1);
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * decodeOut.T);  // ideal gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) * cv2 * decodeOut.T);  // ideal gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
    PetscReal T = decodeOut.T;  // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    PetscReal cv_mix = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    // mixed isothermal sound speed
    PetscReal at_mix = PetscSqrtReal(1 / (w1 + w2)) / density;
    PetscReal Gamma = (w1 * cv1 * (gamma1 - 1) * rho1 + w2 * cv2 * (gamma2 - 1) * rho2) / ((w1 + w2) * cv_mix * density);
    // mixed isentropic sound speed
    *a = PetscSqrtReal(PetscSqr(at_mix) + PetscSqr(Gamma) * cv_mix * T);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundFunctionGasLiquid(const PetscReal *conserved, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleGasStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * decodeOut.T);                                                                   // ideal gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * decodeOut.T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
    PetscReal T = decodeOut.T;  // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    PetscReal cv_mix = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    // mixed isothermal sound speed
    PetscReal at_mix = PetscSqrtReal(1 / (w1 + w2)) / density;
    PetscReal Gamma = (w1 * cv1 * (gamma1 - 1) * rho1 + w2 * cv2 * (gamma2 - 1) * rho2) / ((w1 + w2) * cv_mix * density);
    // mixed isentropic sound speed
    *a = PetscSqrtReal(PetscSqr(at_mix) + PetscSqr(Gamma) * cv_mix * T);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundFunctionLiquidLiquid(const PetscReal *conserved, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleStiffStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) / functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * decodeOut.T);  // stiffened gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * decodeOut.T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;
    PetscReal T = decodeOut.T;  // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    PetscReal cv_mix = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    // mixed isothermal sound speed
    PetscReal at_mix = PetscSqrtReal(1 / (w1 + w2)) / density;
    PetscReal Gamma = (w1 * cv1 * (gamma1 - 1) * rho1 + w2 * cv2 * (gamma2 - 1) * rho2) / ((w1 + w2) * cv_mix * density);
    // mixed isentropic sound speed
    *a = PetscSqrtReal(PetscSqr(at_mix) + PetscSqr(Gamma) * cv_mix * T);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundTemperatureFunctionGasGas(const PetscReal *conserved, PetscReal T, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // same as speedOfSoundFunction, isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleGasGasDecode(functionContext->dim, &decodeIn, &decodeOut);
    // isothermal sound speeds
    cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
    cv2 = functionContext->parameters.rGas2 / (functionContext->parameters.gamma2 - 1);
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * T);  // ideal gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) * cv2 * T);  // ideal gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;  // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    PetscReal cv_mix = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    // mixed isothermal sound speed
    PetscReal at_mix = PetscSqrtReal(1 / (w1 + w2)) / density;
    PetscReal Gamma = (w1 * cv1 * (gamma1 - 1) * rho1 + w2 * cv2 * (gamma2 - 1) * rho2) / ((w1 + w2) * cv_mix * density);
    // mixed isentropic sound speed
    *a = PetscSqrtReal(PetscSqr(at_mix) + PetscSqr(Gamma) * cv_mix * T);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundTemperatureFunctionGasLiquid(const PetscReal *conserved, PetscReal T, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // same as speedOfSoundFunction, isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleGasStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.rGas1 / (functionContext->parameters.gamma1 - 1);
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) * cv1 * T);                                                                   // ideal gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;  // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    PetscReal cv_mix = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    // mixed isothermal sound speed
    PetscReal at_mix = PetscSqrtReal(1 / (w1 + w2)) / density;
    PetscReal Gamma = (w1 * cv1 * (gamma1 - 1) * rho1 + w2 * cv2 * (gamma2 - 1) * rho2) / ((w1 + w2) * cv_mix * density);
    // mixed isentropic sound speed
    *a = PetscSqrtReal(PetscSqr(at_mix) + PetscSqr(Gamma) * cv_mix * T);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TwoPhase::SpeedOfSoundTemperatureFunctionLiquidLiquid(const PetscReal *conserved, PetscReal T, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    // same as speedOfSoundFunction, isentropic sound speed a_mix
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // get kinetic energy for internal energy calculation
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // compute internal energy
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    // simple decode to compute pressure
    DecodeOut decodeOut;
    DecodeIn decodeIn;
    decodeIn.alpha = conserved[functionContext->volumeFractionOffset];
    decodeIn.alphaRho1 = conserved[functionContext->densityVFOffset];
    decodeIn.rho = density;
    decodeIn.e = internalEnergy;
    decodeIn.parameters = functionContext->parameters;

    PetscReal cv1, cv2, at1, at2;  // initialize variables for at_mix

    SimpleStiffStiffDecode(functionContext->dim, &decodeIn, &decodeOut);
    cv1 = functionContext->parameters.Cp1 / functionContext->parameters.gamma1;
    cv2 = functionContext->parameters.Cp2 / functionContext->parameters.gamma2;
    at1 = PetscSqrtReal((functionContext->parameters.gamma1 - 1) / functionContext->parameters.gamma1 * functionContext->parameters.Cp1 * T);  // stiffened gas eos
    at2 = PetscSqrtReal((functionContext->parameters.gamma2 - 1) / functionContext->parameters.gamma2 * functionContext->parameters.Cp2 * T);  // stiffened gas eos

    PetscReal rho1 = decodeOut.rho1;
    PetscReal rho2 = decodeOut.rho2;  // [rho1, rho2, e1, e2, p, T]
    PetscReal gamma1 = functionContext->parameters.gamma1;
    PetscReal gamma2 = functionContext->parameters.gamma2;
    PetscReal Y1 = conserved[functionContext->densityVFOffset] / density;
    PetscReal Y2 = (density - conserved[functionContext->densityVFOffset]) / density;

    // mixed specific heat constant volume
    PetscReal w1 = Y1 / PetscSqr(rho1 * at1);
    PetscReal w2 = Y2 / PetscSqr(rho2 * at2);
    PetscReal cv_mix = Y1 * cv1 + Y2 * cv2 + (w1 * w2) / (w1 + w2) * PetscSqr(cv1 * (gamma1 - 1) * rho1 - cv2 * (gamma2 - 1) * rho2) * T;
    // mixed isothermal sound speed
    PetscReal at_mix = PetscSqrtReal(1 / (w1 + w2)) / density;
    PetscReal Gamma = (w1 * cv1 * (gamma1 - 1) * rho1 + w2 * cv2 * (gamma2 - 1) * rho2) / ((w1 + w2) * cv_mix * density);
    // mixed isentropic sound speed
    *a = PetscSqrtReal(PetscSqr(at_mix) + PetscSqr(Gamma) * cv_mix * T);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TwoPhase::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    for (PetscInt s = 0; s < parameters.numberSpecies1 + parameters.numberSpecies2; s++) {
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
