#include "soot.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/petscUtilities.hpp"

ablate::finiteVolume::processes::Soot::Soot(const std::shared_ptr<eos::EOS>& eosIn, const std::shared_ptr<parameters::Parameters>& options, double thresholdTemperature)
    : eos(std::dynamic_pointer_cast<eos::TChem>(eosIn)), thresholdTemperature(thresholdTemperature) {
    // make sure that the eos is set
    if (!std::dynamic_pointer_cast<eos::TChem>(eosIn)) {
        throw std::invalid_argument("ablate::finiteVolume::processes::Soot only accepts EOS of type eos::TChem");
    }

    // Set the options if provided
    if (options) {
        PetscOptionsCreate(&petscOptions) >> utilities::PetscUtilities::checkError;
        options->Fill(petscOptions);
    }

    // Create a vector and mat for local ode calculation
    VecCreateSeq(PETSC_COMM_SELF, TotalEquations, &pointData) >> utilities::PetscUtilities::checkError;
    MatCreateSeqDense(PETSC_COMM_SELF, TotalEquations, TotalEquations, nullptr, &pointJacobian) >> utilities::PetscUtilities::checkError;
    MatSetFromOptions(pointJacobian) >> utilities::PetscUtilities::checkError;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Create timestepping solver context
          - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    TSCreate(PETSC_COMM_SELF, &pointTs) >> utilities::PetscUtilities::checkError;
    PetscObjectSetOptions((PetscObject)pointTs, petscOptions) >> utilities::PetscUtilities::checkError;
    TSSetType(pointTs, TSARKIMEX) >> utilities::PetscUtilities::checkError;
    TSARKIMEXSetFullyImplicit(pointTs, PETSC_TRUE) >> utilities::PetscUtilities::checkError;
    TSARKIMEXSetType(pointTs, TSARKIMEX4) >> utilities::PetscUtilities::checkError;
    TSSetRHSFunction(pointTs, nullptr, SinglePointSootChemistryRHS, &pointInformation) >> utilities::PetscUtilities::checkError;
    TSSetExactFinalTime(pointTs, TS_EXACTFINALTIME_MATCHSTEP) >> utilities::PetscUtilities::checkError;

    // set the adapting control
    TSSetSolution(pointTs, pointData) >> utilities::PetscUtilities::checkError;
    TSSetTimeStep(pointTs, dtInitDefault) >> utilities::PetscUtilities::checkError;
    TSAdapt adapt;
    TSGetAdapt(pointTs, &adapt) >> utilities::PetscUtilities::checkError;
    TSAdaptSetStepLimits(adapt, 1e-12, 1E-4) >> utilities::PetscUtilities::checkError; /* Also available with -ts_adapt_dt_min/-ts_adapt_dt_max */
    TSSetMaxSNESFailures(pointTs, -1) >> utilities::PetscUtilities::checkError;        /* Retry step an unlimited number of times */
    TSSetFromOptions(pointTs) >> utilities::PetscUtilities::checkError;
    TSGetTimeStep(pointTs, &dtInit) >> utilities::PetscUtilities::checkError;
}
ablate::finiteVolume::processes::Soot::~Soot() {
    if (sourceDm) {
        DMDestroy(&sourceDm) >> utilities::PetscUtilities::checkError;
    }
    if (sourceVec) {
        VecDestroy(&sourceVec) >> utilities::PetscUtilities::checkError;
    }
    if (petscOptions) {
        utilities::PetscUtilities::PetscOptionsDestroyAndCheck("TChemReactions", &petscOptions);
    }
    if (pointTs) {
        TSDestroy(&pointTs) >> utilities::PetscUtilities::checkError;
    }
    if (pointData) {
        VecDestroy(&pointData) >> utilities::PetscUtilities::checkError;
    }
    if (pointJacobian) {
        MatDestroy(&pointJacobian) >> utilities::PetscUtilities::checkError;
    }
}

void ablate::finiteVolume::processes::Soot::Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    Process::Initialize(flow);

    // Create a copy of the dm for the solver
    DM coordDM;
    DMGetCoordinateDM(flow.GetSubDomain().GetDM(), &coordDM) >> utilities::PetscUtilities::checkError;
    DMClone(flow.GetSubDomain().GetDM(), &sourceDm) >> utilities::PetscUtilities::checkError;
    DMSetCoordinateDM(sourceDm, coordDM) >> utilities::PetscUtilities::checkError;

    // Setup the unknown field in the dm.  This is a single field that holds sources for rho, rho*E, rho*U, (rho*V, rho*W), Yi, Y1+1, Y1+n
    PetscFV fvm;
    PetscFVCreate(PetscObjectComm((PetscObject)sourceDm), &fvm) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)fvm, "chemistrySource") >> utilities::PetscUtilities::checkError;
    PetscFVSetFromOptions(fvm) >> utilities::PetscUtilities::checkError;
    PetscFVSetNumComponents(fvm, TotalEquations) >> utilities::PetscUtilities::checkError;

    // Only define the new field over the region used by this solver
    DMLabel regionLabel = nullptr;
    if (auto region = flow.GetRegion()) {
        DMGetLabel(sourceDm, region->GetName().c_str(), &regionLabel);
    }
    DMAddField(sourceDm, regionLabel, (PetscObject)fvm) >> utilities::PetscUtilities::checkError;
    PetscFVDestroy(&fvm) >> utilities::PetscUtilities::checkError;

    // create a vector to hold the source terms
    DMCreateLocalVector(sourceDm, &sourceVec) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(sourceVec) >> utilities::PetscUtilities::checkError;

    // Before each step, compute the source term over the entire dt
    flow.RegisterPreRHSFunction(ComputeSootChemistryPreStep, this);

    // Add the rhs point function for the source
    flow.RegisterRHSFunction(AddSootChemistrySourceToFlow, this);

    // Locate each species in the
    const auto& flowDensityYiId = flow.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD);
    auto mw = eos->GetSpeciesMolecularMass();
    for (int s = 0; s < TOTAL_ODE_SPECIES; ++s) {
        pointInformation.speciesOffset[s] = (PetscInt)flowDensityYiId.ComponentOffset(OdeSpeciesNames[s]);
        pointInformation.speciesIndex[s] = (PetscInt)flowDensityYiId.ComponentIndex(OdeSpeciesNames[s]);
        pointInformation.enthalpyOfFormation[s] = eos->GetEnthalpyOfFormation(OdeSpeciesNames[s]);
        pointInformation.mw[s] = mw[OdeSpeciesNames[s]];
    }

    // size up the yiScratch
    pointInformation.yiScratch.resize(eos->GetSpeciesVariables().size());
    pointInformation.speciesSensibleEnthalpyScratch.resize(eos->GetSpeciesVariables().size());

    // create a mock field, where density/euler as zero offset.  This only works because we know we are using tchem
    std::vector<domain::Field> mockFields = {domain::Field{.name = ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD,
                                                           .numberComponents = 1,
                                                           .components = {"density"},
                                                           .id = -1,
                                                           .subId = -1,
                                                           .offset = 0,
                                                           .location = domain::FieldLocation::SOL,
                                                           .type = domain::FieldType::FVM,
                                                           .tags = {}}};

    // Get the functions to compute thermodynamic properties
    pointInformation.specificHeatConstantVolumeFunction = eos->GetThermodynamicTemperatureMassFractionFunction(eos::ThermodynamicProperty::SpecificHeatConstantVolume, mockFields);
    pointInformation.speciesSensibleEnthalpyFunction = eos->GetThermodynamicTemperatureMassFractionFunction(eos::ThermodynamicProperty::SpeciesSensibleEnthalpy, mockFields);
}

PetscErrorCode ablate::finiteVolume::processes::Soot::ComputeSootChemistryPreStep(ablate::finiteVolume::FiniteVolumeSolver& flow, TS flowTs, PetscReal time, bool initialStage, Vec locX, void* ctx) {
    PetscFunctionBeginUser;
    if (!initialStage) {
        PetscFunctionReturn(0);
    }
    auto soot = (ablate::finiteVolume::processes::Soot*)ctx;

    // Get the valid cell range over this region
    domain::Range cellRange;
    flow.GetCellRange(cellRange);

    // get the dim
    PetscInt dim;
    PetscCall(DMGetDimension(flow.GetSubDomain().GetDM(), &dim));

    // store the current dt
    PetscReal dt;
    PetscCall(TSGetTimeStep(flowTs, &dt));

    // get access to the underlying data for the flow
    const auto& flowEulerId = flow.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    const auto& flowDensityYiId = flow.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD);
    const auto& flowDensityProgressId = flow.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD);

    // get the flowSolution from the ts
    const PetscScalar* solutionArray;
    PetscCall(VecGetArrayRead(locX, &solutionArray));

    // get the aux array for temperature
    const auto& temperatureField = flow.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD);
    auto temperatureDm = flow.GetSubDomain().GetFieldDM(temperatureField);
    auto temperatureVec = flow.GetSubDomain().GetVec(temperatureField);
    const PetscScalar* temperatureArray;
    PetscCall(VecGetArrayRead(temperatureVec, &temperatureArray));

    // Get access to the chemistry source.  This is sized for euler + nspec
    PetscScalar* sourceArray;
    PetscCall(VecGetArray(soot->sourceVec, &sourceArray));

    // get an easy reference to the point information
    auto& pointInformation = soot->pointInformation;

    // March over each cell
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cellRange.GetPoint(c);

        // Get the current state variables for this cell
        const PetscScalar* conserved = nullptr;
        PetscCall(DMPlexPointGlobalRead(flow.GetSubDomain().GetDM(), cell, solutionArray, &conserved));

        // If a real cell (not ghost)
        if (conserved) {
            // store the data for the chemistry ts (T, Yi...)
            PetscReal* temperature;
            PetscCall(DMPlexPointLocalFieldRead(temperatureDm, cell, temperatureField.id, temperatureArray, &temperature));

            if (*temperature > soot->thresholdTemperature) {
                // get access to the point ode solver
                PetscScalar* pointArray;
                PetscCall(VecGetArray(soot->pointData, &pointArray));
                PetscReal density = conserved[flowEulerId.offset + ablate::finiteVolume::CompressibleFlowFields::RHO];
                pointInformation.currentDensity = density;
                pointArray[ODE_T] = *temperature;
                pointArray[ODE_NDD] = conserved[flowDensityProgressId.offset] / density / NddScaling;

                // Fill the yi scratch
                for (PetscInt s = 0; s < flowDensityYiId.numberComponents; s++) {
                    pointInformation.yiScratch[s] = PetscMin(PetscMax(0.0, conserved[flowDensityYiId.offset + s] / density), 1.0);
                }

                for (std::size_t s = 0; s < TOTAL_ODE_SPECIES; s++) {
                    pointArray[s] = pointInformation.yiScratch[pointInformation.speciesIndex[s]];
                }

                // Compute the Cv,mix
                PetscCall(VecRestoreArray(soot->pointData, &pointArray));

                // Do a soft reset on the ode solver
                PetscCall(TSSetTime(soot->pointTs, time));
                PetscCall(TSSetMaxTime(soot->pointTs, time + dt));
                PetscCall(TSSetTimeStep(soot->pointTs, soot->dtInit));
                PetscCall(TSSetStepNumber(soot->pointTs, 0));

                // solver for this point
                PetscCall(TSSolve(soot->pointTs, soot->pointData));

                // Use the updated values to compute the source terms for euler and species transport
                PetscScalar* fieldSource;
                PetscCall(DMPlexPointLocalRef(soot->sourceDm, cell, sourceArray, &fieldSource));

                // get the array data again
                PetscCall(VecGetArray(soot->pointData, &pointArray));

                // store the computed source terms
                fieldSource[ODE_T] = 0.0;
                for (PetscInt s = 0; s < TOTAL_ODE_SPECIES; ++s) {
                    fieldSource[ODE_T] += (conserved[pointInformation.speciesOffset[s]] / density - pointArray[s]) * pointInformation.enthalpyOfFormation[s];
                    fieldSource[s] = pointArray[s] - conserved[pointInformation.speciesOffset[s]] / density;
                }
                // Add in the source term for the change in ndd
                fieldSource[ODE_NDD] = pointArray[ODE_NDD] * NddScaling - conserved[flowDensityProgressId.offset] / density;

                // Now scale everything by density/dt
                for (PetscInt i = 0; i < TotalEquations; i++) {
                    // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
                    fieldSource[i] *= density / dt;
                }

                PetscCall(VecRestoreArray(soot->pointData, &pointArray));
            } else {
                // Use the updated values to compute the source terms for euler and species transport
                PetscScalar* fieldSource;
                PetscCall(DMPlexPointLocalRef(soot->sourceDm, cell, sourceArray, &fieldSource));
                PetscArrayzero(fieldSource, TotalEquations);
            }
        }
    }

    // cleanup
    flow.RestoreRange(cellRange);
    PetscCall(VecRestoreArray(soot->sourceVec, &sourceArray));
    PetscCall(VecRestoreArrayRead(temperatureVec, &temperatureArray));
    PetscCall(VecRestoreArrayRead(locX, &solutionArray));

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::Soot::AddSootChemistrySourceToFlow(const ablate::finiteVolume::FiniteVolumeSolver& solver, DM dm, PetscReal time, Vec xVec, Vec fVec, void* ctx) {
    PetscFunctionBeginUser;
    auto soot = (ablate::finiteVolume::processes::Soot*)ctx;

    // get the cell range
    domain::Range cellRange;
    solver.GetCellRange(cellRange);

    // Get the offsets
    const auto& flowEulerId = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    const auto& flowDensityProgressId = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD);

    // get access to the fArray
    PetscScalar* fArray;
    PetscCall(VecGetArray(fVec, &fArray));

    // get access to the source array in the solver
    const PetscScalar* sourceArray;
    PetscCall(VecGetArrayRead(soot->sourceVec, &sourceArray));

    // get the dm
    // March over each cell
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cellRange.GetPoint(c);

        // read the global f
        PetscScalar* rhs;
        PetscCall(DMPlexPointLocalRef(dm, cell, fArray, &rhs));

        // if a real cell
        if (rhs) {
            // read the source from the local calc
            const PetscScalar* source;
            PetscCall(DMPlexPointLocalRead(soot->sourceDm, cell, sourceArray, &source));

            // add the yi sources
            for (PetscInt s = 0; s < TOTAL_ODE_SPECIES; ++s) {
                rhs[soot->pointInformation.speciesOffset[s]] += source[s];
            }
            // Add in the energy source, it was stored in the temperature location
            rhs[flowEulerId.offset + ablate::finiteVolume::CompressibleFlowFields::RHOE] += source[ODE_T];
            rhs[flowDensityProgressId.offset] += source[ODE_NDD];
        }
    }

    solver.RestoreRange(cellRange);
    PetscCall(VecRestoreArray(fVec, &fArray));
    PetscCall(VecGetArrayRead(soot->sourceVec, &sourceArray));

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::finiteVolume::processes::Soot::SinglePointSootChemistryRHS(TS ts, PetscReal t, Vec xVec, Vec fVec, void* ctx) {
    PetscFunctionBeginUser;
    auto pointInfo = (OdePointInformation*)ctx;

    // extract the read/write arrays
    PetscScalar* xArray;
    PetscCall(VecGetArray(xVec, &xArray));
    PetscCall(VecZeroEntries(fVec));
    PetscScalar* fArray;
    PetscCall(VecGetArray(fVec, &fArray));

    // copy over the updated to the scratch variable for now
    for (std::size_t s = 0; s < TOTAL_ODE_SPECIES; s++) {
        xArray[s] = PetscMax(PetscMin(xArray[s], 1.0), 0.0);
        pointInfo->yiScratch[pointInfo->speciesIndex[s]] = xArray[s];
    }
    xArray[ODE_T] = PetscMax(xArray[ODE_T], 0);
    xArray[ODE_NDD] = PetscMax(xArray[ODE_NDD], 0);

    // Add in the Soot Reaction Sources
    real_type SVF = xArray[C_s] * pointInfo->currentDensity / solidCarbonDensity;

    // Total S.A. of soot / unit volume
    PetscReal SA_V = calculateSurfaceArea_V(xArray[C_s], xArray[ODE_NDD], pointInfo->currentDensity);

    // Need the Concentrations of C2H2, O2, O, and OH
    // It is unclear in the formulations of the Reaction Rates whether to use to concentration in regards to the total mixture or just the gas phace, There is a difference due to the density relation
    //-> For now we will use the concentration to be the concentration in the gas phase as it makes more physical sense
    PetscReal C2H2Conc = pointInfo->currentDensity * PetscMax(0, xArray[C2H2]) / pointInfo->mw[C2H2];
    PetscReal O2Conc = pointInfo->currentDensity * PetscMax(0, xArray[O2]) / pointInfo->mw[O2];
    PetscReal OConc = pointInfo->currentDensity * PetscMax(0, xArray[O]) / pointInfo->mw[O];
    PetscReal OHConc = pointInfo->currentDensity * PetscMax(0, xArray[OH]) / pointInfo->mw[OH];

    // Now plug in and solve the Nucleation, Surface Growth, Agglomeration, and Oxidation sources
    PetscReal NucRate = calculateNucleationReactionRate(xArray[ODE_T], C2H2Conc, SVF);
    PetscReal SGRate = calculateSurfaceGrowthReactionRate(xArray[ODE_T], C2H2Conc, SA_V);

    PetscReal AggRate = calculateAgglomerationRate(xArray[C_s], xArray[ODE_NDD], xArray[ODE_T], pointInfo->currentDensity);
    PetscReal O2OxRate = calculateO2OxidationRate(xArray[C_s], xArray[ODE_NDD], O2Conc, pointInfo->currentDensity, xArray[ODE_T], SA_V);
    PetscReal OOxRate = calculateOOxidationRate(OConc, xArray[ODE_T], SA_V, SVF);
    PetscReal OHOxRate = calculateOHOxidationRate(OHConc, xArray[ODE_T], SA_V, SVF);

    // Now Add these rates correctly to the appropriate species sources (solving Yidot, i.e. also have to divide by the total density.
    // Keep in mind all these rates are kmol/m^3, need to convert to kg/m^3 for each appropriate species as well!
    PetscReal O_totDens = 1. / pointInfo->currentDensity;
    // C2H2 (Loss from Nucleation and Surface Growth)
    fArray[C2H2] += O_totDens * pointInfo->mw[C2H2] * (-NucRate - SGRate);
    // O ( Loss from O Oxidation)
    fArray[O] += O_totDens * pointInfo->mw[O] * (-OOxRate);
    // O2 (Loss from O2 Oxidation)
    fArray[O2] += O_totDens * pointInfo->mw[O2] * (-.5 * O2OxRate);
    // OH ( Loss from OH Oxidation)
    fArray[OH] += O_totDens * pointInfo->mw[OH] * (-OHOxRate);
    // CO ( Generation From All Oxidations)
    fArray[CO] += O_totDens * pointInfo->mw[CO] * (OHOxRate + O2OxRate + OOxRate);
    // H2 ( Generation From Nucleation and SG)
    fArray[H2] += O_totDens * pointInfo->mw[H2] * (NucRate + SGRate);
    // H (Generation from OH Oxidation)
    fArray[H] += O_totDens * pointInfo->mw[H] * (OHOxRate);

    // Now Onto The Solid Carbon and Ndd source terms
    // SC ( generation from Surface growth and Nucleation and loss from all oxidation's)
    fArray[C_s] += O_totDens * pointInfo->mw[C_s] * (2 * (NucRate + SGRate) - O2OxRate - OOxRate - OHOxRate);
    fArray[ODE_NDD] += O_totDens * (NdNuclationConversionTerm * NucRate - AggRate);
    fArray[ODE_NDD] /= NddScaling;

    // compute the specific heat at constant volume.  We set this up to allow density to be the only conserved
    PetscReal cv;
    pointInfo->specificHeatConstantVolumeFunction.function(&(pointInfo->currentDensity), pointInfo->yiScratch.data(), xArray[ODE_T], &cv, pointInfo->specificHeatConstantVolumeFunction.context.get());

    // compute the speciesSensibleEnthalpy and turn into internal energy
    pointInfo->speciesSensibleEnthalpyFunction.function(
        &(pointInfo->currentDensity), pointInfo->yiScratch.data(), xArray[ODE_T], pointInfo->speciesSensibleEnthalpyScratch.data(), pointInfo->speciesSensibleEnthalpyFunction.context.get());

    // compute the temperature source term
    for (std::size_t s = 0; s < TOTAL_ODE_SPECIES; s++) {
        fArray[ODE_T] += fArray[s] * (pointInfo->speciesSensibleEnthalpyScratch[pointInfo->speciesIndex[s]] + pointInfo->enthalpyOfFormation[s] - RUNIV * 1.0e3 / pointInfo->mw[s]);
    }
    fArray[ODE_T] /= -cv;

    PetscCall(VecRestoreArray(xVec, &xArray));
    PetscCall(VecRestoreArray(fVec, &fArray));
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::Soot, "Soot only reactions", ARG(ablate::eos::EOS, "eos", "the tChem eos"),
         OPT(ablate::parameters::Parameters, "options", "any PETSc options for the chemistry ts"),
         OPT(double, "thresholdTemperature", "set a minimum temperature for the chemical kinetics ode integration"));
