#include "pressureGradientScaling.hpp"
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/flowProcess.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::finiteVolume::processes::PressureGradientScaling::PressureGradientScaling(std::shared_ptr<eos::EOS> eos, double alphaInit, double domainLength, double maxAlphaAllowedIn,
                                                                                  double maxDeltaPressureFacIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : eos(std::move(eos)),
      maxAlphaAllowed(maxAlphaAllowedIn > 1.0 ? maxAlphaAllowedIn : 100.0),
      maxDeltaPressureFac(maxDeltaPressureFacIn > 0.0 ? maxDeltaPressureFacIn : 0.05),
      domainLength(domainLength),
      log(std::move(log)),
      alpha(alphaInit) {}

PetscErrorCode ablate::finiteVolume::processes::PressureGradientScaling::UpdatePreconditioner(TS flowTs, ablate::solver::Solver &flow) {
    PetscFunctionBeginUser;

    // Compute global maximum valuesfluxCalculator
    PetscReal pAvgLocal = 0.e+0;
    PetscReal pMinLocal = PETSC_MAX_REAL;
    PetscReal pMaxLocal = 0.e+0;
    PetscReal cMaxLocal = 0.e+0;
    PetscReal machMaxLocal = 0.e+0;
    PetscReal alphaMaxLocal = PETSC_MAX_REAL;

    PetscInt countLocal = 0;

    // get access to the underlying data for the flow
    const auto &flowEulerId = flow.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD);
    PetscInt dim = flow.GetSubDomain().GetDimensions();

    // get the flowSolution from the ts
    Vec globFlowVec = flow.GetSubDomain().GetSolutionVector();
    const PetscScalar *flowArray;
    PetscCall(VecGetArrayRead(globFlowVec, &flowArray));

    // Get the valid cell range over this region
    solver::Range cellRange;
    flow.GetCellRange(cellRange);

    // get decode state function/context
    eos::ThermodynamicFunction computeTemperature = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, flow.GetSubDomain().GetFields());
    eos::ThermodynamicTemperatureFunction computeInternalEnergy = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, flow.GetSubDomain().GetFields());
    eos::ThermodynamicTemperatureFunction computeSpeedOfSound = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, flow.GetSubDomain().GetFields());
    eos::ThermodynamicTemperatureFunction computePressure = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, flow.GetSubDomain().GetFields());

    // check for ghost nodes
    // check to see if there is a ghost label
    auto dm = flow.GetSubDomain().GetDM();
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    // March over each cell
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        PetscBool boundary;
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, cell, &ghost);
        }
        DMIsBoundaryPoint(dm, cell, &boundary);
        PetscInt numChildren;
        DMPlexGetTreeChildren(dm, cell, &numChildren, nullptr);
        if (ghost >= 0 || boundary || numChildren) {
            continue;
        }

        // Get the current state variables for this cell
        const PetscScalar *conserved = nullptr;
        PetscCall(DMPlexPointGlobalRead(flow.GetSubDomain().GetDM(), cell, flowArray, &conserved));

        // If valid cell
        if (conserved) {
            // Extract values
            PetscReal a;
            PetscReal mach;
            PetscReal p;

            // Decode the state to compute
            PetscReal temperature;
            PetscCall(computeTemperature.function(conserved, &temperature, computeTemperature.context.get()));
            PetscCall(computePressure.function(conserved, temperature, &p, computePressure.context.get()));
            PetscCall(computeSpeedOfSound.function(conserved, temperature, &a, computeSpeedOfSound.context.get()));

            PetscReal density = conserved[flowEulerId.offset + CompressibleFlowFields::RHO];
            PetscReal velMag = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                velMag += PetscSqr(conserved[flowEulerId.offset + CompressibleFlowFields::RHOU + d] / density);
            }
            mach = PetscSqrtReal(velMag) / a;

            // Store the max/min values
            countLocal++;
            pAvgLocal += p;
            pMinLocal = PetscMin(pMinLocal, p);
            pMaxLocal = PetscMax(pMaxLocal, p);
            cMaxLocal = PetscMax(cMaxLocal, a);
            machMaxLocal = PetscMax(machMaxLocal, mach);
            alphaMaxLocal = PetscMin(alphaMaxLocal, maxMachAllowed / mach);
        }
    }

    // return the cell range
    flow.RestoreRange(cellRange);

    // Take the global values
    auto comm = flow.GetSubDomain().GetComm();
    PetscReal sumValues[2] = {pAvgLocal, (PetscReal)countLocal};
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, sumValues, 2, MPIU_REAL, MPIU_SUM, comm));
    PetscReal maxValues[3] = {pMaxLocal, cMaxLocal, machMaxLocal};
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, maxValues, 3, MPIU_REAL, MPIU_MAX, comm));
    PetscReal minValues[2] = {pMinLocal, alphaMaxLocal};
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, minValues, 2, MPIU_REAL, MPIU_MIN, comm));

    PetscReal alphaMax = minValues[1];
    PetscReal cMax = maxValues[1];
    PetscReal pRef = sumValues[0] / (sumValues[1]);
    PetscReal maxDeltaP = PetscMax(PetscAbsReal(maxValues[0] - pRef), PetscAbsReal(minValues[0] - pRef));
    maxMach = maxValues[2];

    // Get the current timeStep from TS
    PetscReal dt;
    PetscCall(TSGetTimeStep(flowTs, &dt));

    // Update alpha
    PetscReal alphaOld = alpha;
    PetscReal term1 = 0.5e+0 * dt * cMax / domainLength;
    alpha = (alpha + term1 * (PetscSqrtReal(maxDeltaPressureFac * pRef / (maxDeltaP + 1E-30)) - 1.0));
    alpha = PetscMin(alpha, (1. + maxAlphaChange) * alphaOld);  // avoid alpha jumping up to quickly if maxdelPfac=0
    alpha = PetscMin(alpha, alphaMax);
    alpha = PetscMin(alpha, maxAlphaAllowed);
    alpha = PetscMax(alpha, 1.e+0);

    // Update log
    if (log) {
        log->Printf("PGS: %g (alpha), %g (maxMach),  %g (maxMach'), %g (maxDeltaP)\n", alpha, maxMach, alpha * maxMach, maxDeltaP);
    }
    PetscFunctionReturn(0);
}

void ablate::finiteVolume::processes::PressureGradientScaling::Setup(ablate::finiteVolume::FiniteVolumeSolver &fv) {
    auto preStep = std::bind(&ablate::finiteVolume::processes::PressureGradientScaling::UpdatePreconditioner, this, std::placeholders::_1, std::placeholders::_2);
    fv.RegisterPreStep(preStep);

    // initialize the log if provided
    if (log) {
        log->Initialize(fv.GetSubDomain().GetComm());
    }
}

void ablate::finiteVolume::processes::PressureGradientScaling::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    SaveKeyValue(viewer, "pressureGradientScalingAlpha", alpha);
    PetscFunctionReturnVoid();
}

void ablate::finiteVolume::processes::PressureGradientScaling::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) { RestoreKeyValue(viewer, "pressureGradientScalingAlpha", alpha); }

#include "registrar.hpp"
REGISTER_DERIVED(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::PressureGradientScaling);
REGISTER_DEFAULT(ablate::finiteVolume::processes::PressureGradientScaling, ablate::finiteVolume::processes::PressureGradientScaling,
                 "Rescales the thermodynamic pressure gradient scaling the acoustic propagation speeds to allow for a larger time step.",
                 ARG(ablate::eos::EOS, "eos", "the equation of state used for the flow"), ARG(double, "alphaInit", "the initial alpha"),
                 ARG(double, "domainLength", "the reference length of the domain"), OPT(double, "maxAlphaAllowed", "the maximum allowed alpha during the simulation (default 100)"),
                 OPT(double, "maxDeltaPressureFac", "max variation from mean pressure (default 0.05)"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));