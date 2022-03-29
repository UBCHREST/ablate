#include "lodiBoundary.hpp"
#include <finiteVolume/processes/evTransport.hpp>
#include <finiteVolume/processes/speciesTransport.hpp>
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"

ablate::boundarySolver::lodi::LODIBoundary::LODIBoundary(std::shared_ptr<eos::EOS> eos, std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling)
    : eos(std::move(eos)), pressureGradientScaling(std::move(pressureGradientScaling)), dims(0), nEqs(0), nSpecEqs(0), nEvEqs(0), eulerId(-1), speciesId(-1), evId(-1) {}

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
                                                          PetscReal velnprm, PetscReal Cprm, const PetscReal *rhoYi, const PetscReal *rhoEV, const PetscReal *sL,
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
    for (int ns = 0; ns < nSpecEqs; ns++) {
        mdFdn[sOff[speciesId] + ns] = -(rhoYi[ns] / rho * d[0] + rho * d[2 + dims + ns]);  // species
    }
    for (int ne = 0; ne < nEvEqs; ne++) {
        mdFdn[sOff[evId] + ne] = -(rhoEV[ne] / rho * d[0] + rho * d[2 + dims + nSpecEqs + ne]);  // extra
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

void ablate::boundarySolver::lodi::LODIBoundary::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    // Compute the number of equations that need to be solve
    dims = bSolver.GetSubDomain().GetDimensions();
    if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::EULER_FIELD)) {
        nEqs += bSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD).numberComponents;

        if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD)) {
            updateTemperatureData.computeTemperatureFunction = eos->GetComputeTemperatureFunction();
            updateTemperatureData.computeTemperatureContext = eos->GetComputeTemperatureContext();
            updateTemperatureData.numberSpecies = (PetscInt)eos->GetSpecies().size();

            if (updateTemperatureData.numberSpecies > 0) {
                // add in aux update variables
                bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::EulerTransport::UpdateAuxTemperatureField,
                                               &updateTemperatureData,
                                               std::vector<std::string>{finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD},
                                               {finiteVolume::CompressibleFlowFields::EULER_FIELD, finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD});
            } else {
                // add in aux update variables
                bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::EulerTransport::UpdateAuxTemperatureField,
                                               &updateTemperatureData,
                                               std::vector<std::string>{finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD},
                                               {finiteVolume::CompressibleFlowFields::EULER_FIELD});
            }

            if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::VELOCITY_FIELD)) {
                bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::EulerTransport::UpdateAuxVelocityField,
                                               nullptr,
                                               std::vector<std::string>{finiteVolume::CompressibleFlowFields::VELOCITY_FIELD},
                                               {finiteVolume::CompressibleFlowFields::EULER_FIELD});
            }
        }

        if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD)) {
            updateTemperatureData.computeTemperatureFunction = eos->GetComputeTemperatureFunction();
            updateTemperatureData.computeTemperatureContext = eos->GetComputeTemperatureContext();
            updateTemperatureData.numberSpecies = (PetscInt)eos->GetSpecies().size();

            if (updateTemperatureData.numberSpecies > 0) {
                // add in aux update variables
                bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::EulerTransport::UpdateAuxTemperatureField,
                                               &updateTemperatureData,
                                               std::vector<std::string>{finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD},
                                               {finiteVolume::CompressibleFlowFields::EULER_FIELD, finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD});
            } else {
                // add in aux update variables
                bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::EulerTransport::UpdateAuxTemperatureField,
                                               &updateTemperatureData,
                                               std::vector<std::string>{finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD},
                                               {finiteVolume::CompressibleFlowFields::EULER_FIELD});
            }
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
    if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::DENSITY_EV_FIELD)) {
        nEvEqs = bSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::DENSITY_EV_FIELD).numberComponents;
        nEqs += nEvEqs;

        // Register an update for the yi field
        bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::EVTransport::UpdateEVField,
                                       &nEvEqs,
                                       std::vector<std::string>{finiteVolume::CompressibleFlowFields::EV_FIELD},
                                       {finiteVolume::CompressibleFlowFields::EULER_FIELD, finiteVolume::CompressibleFlowFields::DENSITY_EV_FIELD});
    }

    // Call Initialize to setup the other needed vars
    Initialize(dims, nEqs, nSpecEqs, nEvEqs);
}

void ablate::boundarySolver::lodi::LODIBoundary::Initialize(PetscInt dimsIn, PetscInt nEqsIn, PetscInt nSpecEqsIn, PetscInt nEvEqsIn) {
    dims = dimsIn;
    nEqs = nEqsIn;
    nSpecEqs = nSpecEqsIn;
    nEvEqs = nEvEqsIn;

    // compute the offsets depending on if there are any ev, species
    PetscInt offset = 0;
    eulerId = offset++;
    fieldNames.push_back(finiteVolume::CompressibleFlowFields::EULER_FIELD);

    if (nSpecEqs > 0) {
        speciesId = offset++;
        fieldNames.push_back(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD);
    }

    if (nEvEqsIn > 0) {
        evId = offset++;
        fieldNames.push_back(finiteVolume::CompressibleFlowFields::DENSITY_EV_FIELD);
    }
}