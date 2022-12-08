#include "lodiBoundary.hpp"
#include <finiteVolume/processes/evTransport.hpp>
#include <finiteVolume/processes/speciesTransport.hpp>
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"

ablate::boundarySolver::lodi::LODIBoundary::LODIBoundary(std::shared_ptr<eos::EOS> eos, std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling)
    : pressureGradientScaling(std::move(pressureGradientScaling)), dims(0), nEqs(0), nSpecEqs(0), nEvEqs(0), eulerId(-1), speciesId(-1), eos(std::move(eos)) {}

void ablate::boundarySolver::lodi::LODIBoundary::GetVelAndCPrims(PetscReal velNorm, PetscReal speedOfSound, PetscReal cp, PetscReal cv, PetscReal &velNormPrim, PetscReal &speedOfSoundPrim) {
    PetscReal alpha2 = 1.0;
    PetscReal ralpha2 = 1.;
    PetscReal fourralpha2 = 4;

    if (pressureGradientScaling) {
        alpha2 = PetscSqr(pressureGradientScaling->GetAlpha());
        ralpha2 = 1.0 / alpha2;
        fourralpha2 = 4.0 / alpha2;
    }

    double gam = cp / cv;
    double gamm1 = gam - 1.e+0;
    double gamp1 = gam + 1.e+0;
    double M2 = velNorm / speedOfSound;
    M2 = M2 * M2;
    velNormPrim = 0.5e+0 * (velNorm * (gamp1 - gamm1 / alpha2));
    double gamm12 = gamm1 * gamm1;
    double tmp = 1.e+0 - ralpha2;
    tmp = tmp * tmp;
    speedOfSoundPrim = 0.5e+0 * (speedOfSound * PetscSqrtReal(gamm12 * tmp * M2 + fourralpha2));
}

void ablate::boundarySolver::lodi::LODIBoundary::GetEigenValues(PetscReal veln, PetscReal c, PetscReal velnprm, PetscReal cprm, PetscReal *lamda) const {
    lamda[0] = velnprm - cprm;
    lamda[1] = veln;
    for (int ndim = 1; ndim < dims; ndim++) {
        lamda[1 + ndim] = veln;
    }
    lamda[1 + dims] = velnprm + cprm;
    for (int ns = 0; ns < nSpecEqs; ns++) {
        lamda[2 + dims + ns] = veln;
    }
    for (int ne = 0; ne < nEvEqs; ne++) {
        lamda[2 + dims + nSpecEqs + ne] = veln;
    }
}

void ablate::boundarySolver::lodi::LODIBoundary::GetmdFdn(const PetscInt sOff[], const PetscReal *vel, PetscReal rho, PetscReal T, PetscReal Cp, PetscReal Cv, PetscReal C, PetscReal Enth,
                                                          PetscReal velnprm, PetscReal Cprm, const PetscReal *conserved, const PetscInt uOff[], const PetscReal *sL,
                                                          const PetscReal transformationMatrix[3][3], PetscReal *mdFdn) const {
    std::vector<PetscScalar> d(nEqs);
    PetscReal alpha2 = 1.0;
    if (pressureGradientScaling) {
        alpha2 = PetscSqr(pressureGradientScaling->GetAlpha());
    }

    auto fac = 0.5e+0 * (sL[0] - sL[1 + dims]) * (velnprm - vel[0]) / Cprm;
    double C2 = C * C;
    d[0] = (sL[1] + 0.5e+0 * (sL[1 + dims] + sL[0]) + fac) / C2;
    d[1] = 0.5e+0 * (sL[1 + dims] + sL[0]) - fac;
    d[2] = 0.5e+0 * (sL[1 + dims] - sL[0]) / rho / Cprm / alpha2;
    for (int ndim = 1; ndim < dims; ndim++) {
        d[2 + ndim] = sL[1 + ndim];
    }
    for (int ns = 0; ns < nSpecEqs; ns++) {
        d[2 + dims + ns] = sL[2 + dims + ns];
    }
    for (int ne = 0; ne < nEvEqs; ne++) {
        d[2 + dims + nSpecEqs + ne] = sL[2 + dims + nSpecEqs + ne];
    }
    mdFdn[sOff[eulerId] + RHO] = -d[0];
    mdFdn[sOff[eulerId] + RHOVELN] = -(vel[0] * d[0] + rho * d[2]);  // Wall normal component momentum, not really rho u
    double KE = vel[0] * vel[0];
    double dvelterm = vel[0] * d[2];
    for (int ndim = 1; ndim < dims; ndim++) {  // Tangential components for momentum
        mdFdn[sOff[eulerId] + RHOVELN + ndim] = -(vel[ndim] * d[0] + rho * d[2 + ndim]);
        KE += vel[ndim] * vel[ndim];
        dvelterm = dvelterm + vel[ndim] * d[2 + ndim];
    }
    KE = 0.5e+0 * KE;
    mdFdn[sOff[eulerId] + RHOE] = -(d[0] * (KE + Enth - Cp * T) + d[1] / (Cp / Cv - 1.e+0 + 1.0E-30) + rho * dvelterm);
    const PetscReal *rhoYi = conserved + uOff[speciesId];
    for (int ns = 0; ns < nSpecEqs; ns++) {
        mdFdn[sOff[speciesId] + ns] = -(rhoYi[ns] / rho * d[0] + rho * d[2 + dims + ns]);  // species
    }

    int ne = 0;
    for (std::size_t ev = 0; ev < evIds.size(); ++ev) {
        const PetscReal *rhoEV = conserved + uOff[evIds[ev]];
        for (PetscInt ec = 0; ec < nEvComps[ev]; ++ec) {
            mdFdn[sOff[evIds[ev]] + ec] = -(rhoEV[ec] / rho * d[0] + rho * d[2 + dims + nSpecEqs + ne]);  // extra
            ne++;
        }
    }

    /*
        map momentum source terms (normal, tangent 1, tangent 2 back to
        physical coordinate system (x-mom, y-mom, z-mom). Note, that the
        normal direction points outward from the domain. The source term for
        the normal component of momentum is therefore for the velocity
        pointing outward from the surface. For ncomp=1 (Cartesian mesh with
        mapped space aligned with physical space) and nside=0,2 & 4 a minus
        one is multiplied by the normal component. For ncomp > 1, the dircos
        data structure is used which is more general but also more expensive.
     */
    PetscReal mdFdntmp[3] = {0.0, 0.0, 0.0};
    utilities::MathUtilities::MultiplyTranspose(dims, transformationMatrix, mdFdn + sOff[eulerId] + RHOVELN, mdFdntmp);
    // Over-write source components
    for (PetscInt nc = 0; nc < dims; nc++) {
        mdFdn[sOff[eulerId] + RHOVELN + nc] = mdFdntmp[nc];
    }
}

void ablate::boundarySolver::lodi::LODIBoundary::Setup(ablate::boundarySolver::BoundarySolver &bSolver) {
    // Compute the number of equations that need to be solved
    dims = bSolver.GetSubDomain().GetDimensions();
    if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::EULER_FIELD)) {
        nEqs += bSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD).numberComponents;

        if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::VELOCITY_FIELD)) {
            bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::NavierStokesTransport::UpdateAuxVelocityField,
                                           nullptr,
                                           std::vector<std::string>{finiteVolume::CompressibleFlowFields::VELOCITY_FIELD},
                                           {finiteVolume::CompressibleFlowFields::EULER_FIELD});
        }

        if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD)) {
            // set decode state functions
            computeTemperatureFunction = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Temperature, bSolver.GetSubDomain().GetFields());
            // add in aux update variables
            bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::NavierStokesTransport::UpdateAuxTemperatureField,
                                           &computeTemperatureFunction,
                                           std::vector<std::string>{finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD},
                                           {});
        }
    }
    if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD)) {
        nSpecEqs = bSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD).numberComponents;
        nEqs += nSpecEqs;

        // Register an update for the yi field
        bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::SpeciesTransport::UpdateAuxMassFractionField,
                                       &nSpecEqs,
                                       std::vector<std::string>{finiteVolume::CompressibleFlowFields::YI_FIELD},
                                       {finiteVolume::CompressibleFlowFields::EULER_FIELD, finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD});

        bSolver.RegisterPostEvaluate(ablate::finiteVolume::processes::SpeciesTransport::NormalizeSpecies);
    }

    for (const auto &evField : bSolver.GetSubDomain().GetFields(domain::FieldLocation::SOL, finiteVolume::CompressibleFlowFields::EV_TAG)) {
        auto &nEvComp = nEvComps.emplace_back(evField.numberComponents);
        nEqs += nEvComp;

        // Register an update for the ev field
        auto evNonConserved = evField.name.substr(finiteVolume::CompressibleFlowFields::CONSERVED.length());
        bSolver.RegisterAuxFieldUpdate(
            ablate::finiteVolume::processes::EVTransport::UpdateEVField, &nEvComp, std::vector<std::string>{evNonConserved}, {finiteVolume::CompressibleFlowFields::EULER_FIELD, evField.name});
    }

    // Call Initialize to setup the other needed vars
    Setup(dims, nEqs, nSpecEqs, nEvComps, bSolver.GetSubDomain().GetFields());
}

void ablate::boundarySolver::lodi::LODIBoundary::Setup(PetscInt dimsIn, PetscInt nEqsIn, PetscInt nSpecEqsIn, std::vector<PetscInt> nEvCompsIn, const std::vector<domain::Field> &fields) {
    dims = dimsIn;
    nEqs = nEqsIn;
    nSpecEqs = nSpecEqsIn;
    if (nEvComps.empty()) {
        nEvComps = nEvCompsIn;
    }

    // compute the idOffsets depending on if there are any ev, species
    PetscInt idOffset = 0;
    eulerId = idOffset++;
    fieldNames.push_back(finiteVolume::CompressibleFlowFields::EULER_FIELD);

    if (nSpecEqs > 0) {
        speciesId = idOffset++;
        fieldNames.push_back(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD);
    }

    // March over each field, assume the ev order is the same as specified in the field
    nEvEqs = 0;
    std::size_t evCount = 0;
    for (const auto &field : fields) {
        if (field.Tagged(finiteVolume::CompressibleFlowFields::EV_TAG)) {
            evIds.push_back(idOffset++);
            fieldNames.push_back(field.name);
            nEvEqs += field.numberComponents;

            // sanity check
            if (nEvComps[evCount++] != field.numberComponents) {
                throw std::invalid_argument("Error setting up LODIBoundary.  The EV component does not match.");
            }
        }
    }

    // set decode state functions
    computeTemperature = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, fields);
    computeSpeedOfSound = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, fields);
    computePressureFromTemperature = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, fields);

    computeSpecificHeatConstantPressure = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpecificHeatConstantPressure, fields);
    computeSpecificHeatConstantVolume = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpecificHeatConstantVolume, fields);
    computeSensibleEnthalpyFunction = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SensibleEnthalpy, fields);
    computePressure = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Pressure, fields);
}