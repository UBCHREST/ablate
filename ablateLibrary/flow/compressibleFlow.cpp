#include "compressibleFlow.hpp"
#include <utilities/mpiError.hpp>
#include "compressibleFlow.h"
#include "flow/fluxDifferencer/ausmFluxDifferencer.hpp"
#include "fvSupport.h"
#include "utilities/petscError.hpp"

static const char* compressibleFlowComponentNames[TOTAL_COMPRESSIBLE_FLOW_COMPONENTS + 1] = {"rho", "rhoE", "rhoU", "rhoV", "rhoW", "unknown"};
static const char* compressibleAuxComponentNames[TOTAL_COMPRESSIBLE_AUX_COMPONENTS + 1] = {"T", "vel", "unknown"};

static PetscErrorCode UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscScalar* conservedValues, PetscScalar* auxField, void* ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE] / density;
    FlowData_CompressibleFlow flowParameters = (FlowData_CompressibleFlow)ctx;
    PetscErrorCode ierr = flowParameters->computeTemperatureFunction(
        dim, density, totalEnergy, conservedValues + RHOU, flowParameters->numberSpecies ? conservedValues + RHOU + dim : NULL, &auxField[T], flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode UpdateAuxVelocityField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscScalar* conservedValues, PetscScalar* auxField, void* ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[RHO];

    for (PetscInt d = 0; d < dim; d++) {
        auxField[d] = conservedValues[RHOU + d] / density;
    }

    PetscFunctionReturn(0);
}

ablate::flow::CompressibleFlow::CompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<parameters::Parameters> parameters,
                                                 std::shared_ptr<fluxDifferencer::FluxDifferencer> fluxDifferencerIn, std::shared_ptr<parameters::Parameters> options,
                                                 std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization,
                                                 std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                 std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolutions)
    : FVFlow(name, mesh, parameters,
             {
                 {.fieldName = "euler", .fieldPrefix = "euler", .components = 2 + mesh->GetDimensions(), .fieldType = FieldType::FV},
                 {
                     .fieldName = "densityYi",
                     .fieldPrefix = "densityYi",
                     .components = (PetscInt)eosIn->GetSpecies().size(),
                     .fieldType = FieldType::FV,
                     .componentNames = eosIn->GetSpecies(),
                 },
                 {.solutionField = false, .fieldName = compressibleAuxComponentNames[T], .fieldPrefix = compressibleAuxComponentNames[T], .components = 1, .fieldType = FieldType::FV},
                 {.solutionField = false, .fieldName = compressibleAuxComponentNames[VEL], .fieldPrefix = compressibleAuxComponentNames[VEL], .components = mesh->GetDimensions(), .fieldType = FieldType::FV}
             }, options, initialization, boundaryConditions, {}, exactSolutions),
      eos(eosIn),
      fluxDifferencer(fluxDifferencerIn == nullptr ? std::make_shared<fluxDifferencer::AusmFluxDifferencer>() : fluxDifferencerIn) {
    // Create a compressibleFlowData
    PetscNew(&compressibleFlowData);

    // Store the required data for the low level c functions
    compressibleFlowData->cfl = parameters->Get<PetscReal>("cfl", 0.5);
    compressibleFlowData->mu = parameters->Get<PetscReal>("mu", 0.0);
    compressibleFlowData->k = parameters->Get<PetscReal>("k", 0.0);

    // set the decode state function
    compressibleFlowData->decodeStateFunction = eos->GetDecodeStateFunction();
    compressibleFlowData->decodeStateFunctionContext = eos->GetDecodeStateContext();
    compressibleFlowData->computeTemperatureFunction = eos->GetComputeTemperatureFunction();
    compressibleFlowData->computeTemperatureContext = eos->GetComputeTemperatureContext();
    compressibleFlowData->numberSpecies = eos->GetSpecies().size();

    // Start problem setup
    PetscDS prob;
    DMGetDS(dm->GetDomain(), &prob) >> checkError;

    // Set up each field
    PetscInt eulerField = 0;
    PetscDSSetContext(prob, eulerField, compressibleFlowData) >> checkError;

    // register the flow fields source terms
    if (eos->GetSpecies().empty()) {
        RegisterRHSFunction(CompressibleFlowComputeEulerFlux, compressibleFlowData, "euler", {"euler"}, {});
    } else {
        RegisterRHSFunction(CompressibleFlowComputeEulerFlux, compressibleFlowData, "euler", {"euler", "densityYi"}, {});
        RegisterRHSFunction(CompressibleFlowSpeciesAdvectionFlux, compressibleFlowData, "densityYi", {"euler", "densityYi"}, {});
    }

    // if there are any coefficients for diffusion, compute diffusion
    if (compressibleFlowData->k || compressibleFlowData->mu) {
        RegisterRHSFunction(CompressibleFlowEulerDiffusion, compressibleFlowData, "euler", {"euler"}, {"T", "vel"});
    }

    // Set the flux calculator solver for each component
    PetscDSSetFromOptions(prob) >> checkError;

    // extract the difference function from fluxDifferencer object
    compressibleFlowData->fluxDifferencer = fluxDifferencer->GetFluxDifferencerFunction();

    // Set the update fields
    RegisterAuxFieldUpdate(UpdateAuxTemperatureField, compressibleFlowData, "T");
    RegisterAuxFieldUpdate(UpdateAuxVelocityField, compressibleFlowData, "vel");

    // PetscErrorCode PetscOptionsGetBool(PetscOptions options,const char pre[],const char name[],PetscBool *ivalue,PetscBool *set)
    compressibleFlowData->automaticTimeStepCalculator = PETSC_TRUE;
    PetscOptionsGetBool(NULL, NULL, "-automaticTimeStepCalculator", &(compressibleFlowData->automaticTimeStepCalculator), NULL);

    auto numberSpecies = compressibleFlowData->numberSpecies;
    //    RegisterPostEvaluate([numberSpecies](auto ts, auto& flow){
    //        Vec solutionVec;
    //        TSGetSolution(ts, &solutionVec) >> checkError;
    //        DM dm;
    //        TSGetDM(ts, &dm) >> checkError;
    //
    //        // March over each species to limit the mass fraction between 0 and 1.  Make the last one equal to the first
    //        PetscScalar* array;
    //        VecGetArray(solutionVec, &array) >>checkError;
    //
    //        // get the field location for yi
    //      PetscInt densityLoc = flow.GetFieldId("euler").value();
    //
    //      PetscInt yiLoc = flow.GetFieldId("densityYi").value();
    //
    //        PetscInt cStart, cEnd;
    //        DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd) >>checkError;
    //
    //        for(PetscInt c = cStart; c < cEnd; c++){
    //            PetscReal *yiArray;
    //            PetscReal *densityArray;
    //            DMPlexPointGlobalFieldRef(dm, c, densityLoc, array, &densityArray) >>checkError;
    //            DMPlexPointGlobalFieldRef(dm, c, yiLoc, array, &yiArray) >>checkError;
    //            if (yiArray) {  // must be real cell and not ghost
    //                PetscScalar sum = 0.0;
    //                for(PetscInt sp = 0; sp < numberSpecies -1; sp ++){
    //                    yiArray[sp] = densityArray[0]*PetscMax(0.0, PetscMin(1.0, yiArray[sp]/densityArray[0] ));
    //                    sum +=yiArray[sp];
    //                }
    //                yiArray[ numberSpecies -1] = densityArray[0] - sum;
    //            }
    //
    //        }
    //
    //        VecRestoreArray(solutionVec, &array) >> checkError;
    //
    //        return 0;
    //    });
}

ablate::flow::CompressibleFlow::~CompressibleFlow() { PetscFree(compressibleFlowData); }

void ablate::flow::CompressibleFlow::ComputeTimeStep(TS ts, ablate::flow::Flow& flow) {
    PetscFunctionBeginUser;
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> checkError;
    Vec v;
    TSGetSolution(ts, &v) >> checkError;

    // Get the flow param
    ablate::flow::CompressibleFlow& compressibleFlow = dynamic_cast<ablate::flow::CompressibleFlow&>(flow);
    FlowData_CompressibleFlow flowParameters = compressibleFlow.compressibleFlowData;

    // Get the fv geom
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> checkError;
    PetscInt cStart, cEnd;
    DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd) >> checkError;
    const PetscScalar* x;
    VecGetArrayRead(v, &x) >> checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx = 2.0 * minCellRadius;

    // Get field location for euler and densityYi
    auto eulerId = flow.GetFieldId("euler").value();
    auto densityYiId = flow.GetFieldId("densityYi").value_or(-1);

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscReal* xc;
        const PetscReal* densityYi = NULL;
        DMPlexPointGlobalFieldRead(dm, c, eulerId, x, &xc) >> checkError;

        if (densityYiId >= 0) {
            DMPlexPointGlobalFieldRead(dm, c, densityYiId, x, &densityYi) >> checkError;
        }

        if (xc) {  // must be real cell and not ghost
            PetscReal rho = xc[RHO];
            PetscReal vel[3];
            for (PetscInt i = 0; i < dim; i++) {
                vel[i] = xc[RHOU + i] / rho;
            }

            // Get the speed of sound from the eos
            PetscReal ie;
            PetscReal a;
            PetscReal p;
            flowParameters->decodeStateFunction(dim, rho, xc[RHOE] / rho, vel, densityYi, &ie, &a, &p, flowParameters->decodeStateFunctionContext) >> checkError;

            PetscReal u = xc[RHOU] / rho;
            PetscReal dt = flowParameters->cfl * dx / (a + PetscAbsReal(u));
            dtMin = PetscMin(dtMin, dt);

            if (PetscIsNanReal(dt)) {
                throw std::runtime_error("Invalid timestep selected for flow");
            }
        }
    }
    PetscInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);

    PetscReal dtMinGlobal;
    MPI_Allreduce(&dtMin, &dtMinGlobal, 1, MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject)ts)) >> checkMpiError;

    TSSetTimeStep(ts, dtMinGlobal) >> checkError;

    if (PetscIsNanReal(dtMinGlobal)) {
        throw std::runtime_error("Invalid timestep selected for flow");
    }

    VecRestoreArrayRead(v, &x) >> checkError;
}

void ablate::flow::CompressibleFlow::CompleteProblemSetup(TS ts) {
    FVFlow::CompleteProblemSetup(ts);

    if (compressibleFlowData->automaticTimeStepCalculator) {
        preStepFunctions.push_back(ComputeTimeStep);
    }
}
void ablate::flow::CompressibleFlow::CompleteFlowInitialization(DM, Vec) {}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::Flow, ablate::flow::CompressibleFlow, "compressible finite volume flow", ARG(std::string, "name", "the name of the flow field"),
         ARG(ablate::mesh::Mesh, "mesh", "the  mesh and discretization"), ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         ARG(ablate::parameters::Parameters, "parameters", "the compressible flow parameters cfl, gamma, etc."),
         OPT(ablate::flow::fluxDifferencer::FluxDifferencer, "fluxDifferencer", "the flux differencer (defaults to AUSM)"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSc"), OPT(std::vector<mathFunctions::FieldSolution>, "initialization", "the flow field initialization"),
         OPT(std::vector<flow::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldSolution>, "exactSolution", "optional exact solutions that can be used for error calculations"));