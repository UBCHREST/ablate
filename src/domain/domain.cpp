#include "domain.hpp"
#include <set>
#include <utility>
#include "solver/solver.hpp"
#include "subDomain.hpp"
#include "utilities/demangler.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

ablate::domain::Domain::Domain(DM dmIn, std::string name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptorsIn, std::vector<std::shared_ptr<modifiers::Modifier>> modifiersIn,
                               const std::shared_ptr<parameters::Parameters>& options, bool setFromOptions)
    : dm(dmIn), name(std::move(name)), comm(PetscObjectComm((PetscObject)dm)), fieldDescriptors(std::move(fieldDescriptorsIn)), solGlobalField(nullptr), modifiers(std::move(modifiersIn)) {
    // if provided, convert options to a petscOptions
    if (options) {
        PetscOptionsCreate(&petscOptions) >> utilities::PetscUtilities::checkError;
        options->Fill(petscOptions);
    }

    // Apply petsc options to the domain
    PetscObjectSetOptions((PetscObject)dm, petscOptions) >> utilities::PetscUtilities::checkError;
    if (setFromOptions) {
        DMSetFromOptions(dm) >> utilities::PetscUtilities::checkError;
    }
    // update the dm with the modifiers
    for (auto& modifier : modifiers) {
        modifier->Modify(dm);
    }

    // register all the solution fields with the DM, store the aux fields for later
    std::vector<std::shared_ptr<FieldDescription>> allAuxFields;
    for (const auto& fieldDescriptor : fieldDescriptors) {
        for (auto& fieldDescription : fieldDescriptor->GetFields()) {
            fieldDescription->DecompressComponents(GetDimensions());
            switch (fieldDescription->location) {
                case FieldLocation::SOL:
                    RegisterField(*fieldDescription);
                    break;
                case FieldLocation::AUX:
                    allAuxFields.push_back(fieldDescription);
                    break;
                default:
                    throw std::invalid_argument("Unknown Field Location for " + fieldDescription->name);
            }
        }
    }

    // Set up the global DS
    DMCreateDS(dm) >> utilities::PetscUtilities::checkError;

    // based upon the ds divisions in the dm, create a subDomain for each
    PetscInt numberDS;
    DMGetNumDS(dm, &numberDS) >> utilities::PetscUtilities::checkError;

    // March over each ds and create a subDomain
    for (PetscInt ds = 0; ds < numberDS; ds++) {
        subDomains.emplace_back(std::make_shared<ablate::domain::SubDomain>(*this, ds, allAuxFields));
    }
}

ablate::domain::Domain::~Domain() {
    // clean up the petsc objects
    if (solGlobalField) {
        VecDestroy(&solGlobalField) >> utilities::PetscUtilities::checkError;
    }
    if (petscOptions) {
        ablate::utilities::PetscUtilities::PetscOptionsDestroyAndCheck("ablate::domain::Domain", &petscOptions);
    }
}

void ablate::domain::Domain::RegisterField(const ablate::domain::FieldDescription& fieldDescription) {
    // make sure that this is a solution field
    if (fieldDescription.location != FieldLocation::SOL) {
        throw std::invalid_argument("The field must be FieldLocation::SOL to be registered with the domain");
    }

    // Look up the label for this field
    DMLabel label = nullptr;
    if (fieldDescription.region) {
        DMGetLabel(dm, fieldDescription.region->GetName().c_str(), &label) >> utilities::PetscUtilities::checkError;
        if (label == nullptr) {
            throw std::invalid_argument("Cannot locate label " + fieldDescription.region->GetName() + " for field " + fieldDescription.name);
        }
    }

    // Create the field and add it with the label
    auto petscField = fieldDescription.CreatePetscField(dm);

    // add to the dm
    DMAddField(dm, label, petscField);

    // Free the petsc after being added
    PetscObjectDestroy(&petscField);

    // Record the field
    fields.push_back(Field::FromFieldDescription(fieldDescription, (PetscInt)fields.size()));
}

PetscInt ablate::domain::Domain::GetDimensions() const noexcept {
    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;
    return dim;
}

void ablate::domain::Domain::CreateStructures() {
    // Setup the solve with the ts
    DMPlexCreateClosureIndex(dm, nullptr) >> utilities::PetscUtilities::checkError;
    DMCreateGlobalVector(dm, &(solGlobalField)) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)solGlobalField, solution_vector_name.c_str()) >> utilities::PetscUtilities::checkError;

    // add the names to each of the components in the dm section
    PetscSection section;
    DMGetLocalSection(dm, &section) >> utilities::PetscUtilities::checkError;
    for (const auto& field : fields) {
        if (field.numberComponents > 1) {
            for (PetscInt c = 0; c < field.numberComponents; c++) {
                PetscSectionSetComponentName(section, field.id, c, field.components[c].c_str()) >> utilities::PetscUtilities::checkError;
            }
        }
    }
}

std::shared_ptr<ablate::domain::SubDomain> ablate::domain::Domain::GetSubDomain(const std::shared_ptr<domain::Region>& region) const {
    // Check to see if there is a label for this region
    if (region) {
        // March over each ds region, and return the subdomain if this region is inside of any subDomain region
        for (const auto& subDomain : subDomains) {
            if (subDomain->InRegion(*region)) {
                return subDomain;
            }
        }
        throw std::runtime_error("Unable to locate subDomain for region " + region->ToString());

    } else {
        // Get the only subDomain
        if (subDomains.size() > 1) {
            throw std::runtime_error("More than one DS was created, the region is expected to be defined.");
        }
        return subDomains.front();
    }
}

void ablate::domain::Domain::InitializeSubDomains(const std::vector<std::shared_ptr<solver::Solver>>& solvers, const std::shared_ptr<ablate::domain::Initializer>& initializations,
                                                  const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& exactSolutions) {
    // determine the number of fields
    for (auto& solver : solvers) {
        solver->Register(GetSubDomain(solver->GetRegion()));
    }

    StartEvent("Domain::Setup");
    // Setup each of the fields
    for (auto& solver : solvers) {
        solver->Setup();
    }
    EndEvent();

    // Create the global structures
    CreateStructures();
    for (auto& subDomain : subDomains) {
        subDomain->CreateSubDomainStructures();
    }

    // set all values to nan to allow for a output check
    if (initializations) {
        VecSet(solGlobalField, NAN) >> utilities::PetscUtilities::checkError;

        // Set the initial conditions for each field specified
        auto initializationsFieldFunctions = initializations->GetFieldFunctions(GetFields());
        ProjectFieldFunctions(initializationsFieldFunctions, solGlobalField);
    }

    // Initialize each solver
    StartEvent("Domain::Initialize");
    for (auto& solver : solvers) {
        solver->Initialize();
    }

    // do a sanity check to make sure that all points were initialized
    if (initializations && CheckFieldValues()) {
        throw std::runtime_error("Field values at points in the domain were not initialized.");
    }

    EndEvent();

    // Set the exact solutions if the field in lives in each subDomain
    for (auto& subDomain : subDomains) {
        subDomain->SetsExactSolutions(exactSolutions);
    }
}
std::vector<std::weak_ptr<ablate::io::Serializable>> ablate::domain::Domain::GetSerializableSubDomains() {
    std::vector<std::weak_ptr<io::Serializable>> serializables;
    for (auto& serializable : subDomains) {
        serializables.push_back(serializable);
    }
    return serializables;
}

void ablate::domain::Domain::ProjectFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& fieldFunctions, Vec globVec, PetscReal time) {
    PetscInt numberFields;
    DMGetNumFields(dm, &numberFields) >> utilities::PetscUtilities::checkError;

    // get a local vector for the work
    Vec locVec;
    DMGetLocalVector(dm, &locVec) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(dm, globVec, INSERT_VALUES, locVec) >> utilities::PetscUtilities::checkError;

    for (auto& fieldFunction : fieldFunctions) {
        // Size up the field projects
        std::vector<mathFunctions::PetscFunction> fieldFunctionsPts(numberFields, nullptr);
        std::vector<void*> fieldContexts(numberFields, nullptr);

        auto fieldId = GetField(fieldFunction->GetName());
        fieldContexts[fieldId.id] = fieldFunction->GetSolutionField().GetContext();
        fieldFunctionsPts[fieldId.id] = fieldFunction->GetSolutionField().GetPetscFunction();

        // Determine where to apply this field
        DMLabel fieldLabel = nullptr;
        PetscInt fieldValue = 0;
        if (const auto& region = fieldFunction->GetRegion()) {
            fieldValue = region->GetValue();
            DMGetLabel(dm, region->GetName().c_str(), &fieldLabel) >> utilities::PetscUtilities::checkError;
        } else {
            PetscObject fieldTemp;
            DMGetField(dm, fieldId.id, &fieldLabel, &fieldTemp) >> utilities::PetscUtilities::checkError;
            if (fieldLabel) {
                fieldValue = 1;  // this is temporary until petsc allows fields to be defined with values beside 1
            }
        }

        // Note the global DMProjectFunctionLabel can't be used because it overwrites unwritten values.
        // Project this field
        if (fieldLabel) {
            // make sure that some of this field exists here
            IS regionIS;
            DMLabelGetStratumIS(fieldLabel, fieldValue, &regionIS) >> utilities::PetscUtilities::checkError;

            if (regionIS) {
                DMProjectFunctionLabelLocal(dm, time, fieldLabel, 1, &fieldValue, -1, nullptr, fieldFunctionsPts.data(), fieldContexts.data(), INSERT_VALUES, locVec) >>
                    utilities::PetscUtilities::checkError;
                ISDestroy(&regionIS) >> utilities::PetscUtilities::checkError;
            }
        } else {
            DMProjectFunctionLocalMixedCells(dm, time, fieldFunctionsPts.data(), fieldContexts.data(), INSERT_VALUES, locVec) >> utilities::PetscUtilities::checkError;
        }
    }

    // push the results back to the global vector
    DMLocalToGlobal(dm, locVec, INSERT_VALUES, globVec) >> utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(dm, &locVec) >> utilities::PetscUtilities::checkError;
}

bool ablate::domain::Domain::CheckFieldValues(Vec globSourceVector) {
    // create a set of points that have failed
    std::set<PetscInt> failedPoints;

    // Get the solution and aux info
    Vec solutionVec = GetSolutionVector();
    const PetscScalar* solutionArray;
    VecGetArrayRead(solutionVec, &solutionArray) >> utilities::PetscUtilities::checkError;

    // march over point in the domain
    PetscInt pStart, pEnd;
    DMPlexGetChart(GetDM(), &pStart, &pEnd) >> utilities::PetscUtilities::checkError;

    // get the global section
    PetscSection globalSection;
    DMGetGlobalSection(GetDM(), &globalSection) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt p = pStart; p < pEnd; ++p) {
        const PetscScalar* solutionAtP = nullptr;
        DMPlexPointGlobalRead(GetDM(), p, solutionArray, &solutionAtP) >> utilities::PetscUtilities::checkError;

        // check each scalar for nan/inf
        if (solutionAtP) {
            PetscInt dof;
            PetscSectionGetDof(globalSection, p, &dof);
            PetscInt cdof;
            PetscSectionGetConstraintDof(globalSection, p, &cdof);
            for (PetscInt m = 0; m < (dof - cdof); ++m) {
                if (PetscIsInfOrNanScalar(solutionAtP[m])) {
                    failedPoints.insert(p);
                }
            }
        }
    }

    // If the global source vector is provided also check it
    const PetscScalar* sourceArray;
    DM sourceDM = nullptr;
    if (globSourceVector) {
        VecGetArrayRead(globSourceVector, &sourceArray) >> utilities::PetscUtilities::checkError;
        VecGetDM(globSourceVector, &sourceDM) >> utilities::PetscUtilities::checkError;

        // get the global section
        PetscSection sourceSection;
        DMGetLocalSection(sourceDM, &sourceSection) >> utilities::PetscUtilities::checkError;

        for (PetscInt p = pStart; p < pEnd; ++p) {
            const PetscScalar* sourceAtP = nullptr;
            DMPlexPointGlobalRead(sourceDM, p, sourceArray, &sourceAtP) >> utilities::PetscUtilities::checkError;

            // check each scalar for nan/inf
            if (sourceAtP) {
                PetscInt dof;
                PetscSectionGetDof(sourceSection, p, &dof);
                PetscInt cdof;
                PetscSectionGetConstraintDof(globalSection, p, &cdof);
                for (PetscInt m = 0; m < (dof - cdof); ++m) {
                    if (PetscIsInfOrNanScalar(sourceAtP[m])) {
                        failedPoints.insert(p);
                    }
                }
            }
        }
    }

    // do a global check
    auto localFailedPoints = (PetscMPIInt)failedPoints.size();
    PetscMPIInt globalFailedPoints;
    MPI_Allreduce(&localFailedPoints, &globalFailedPoints, 1, MPI_INT, MPI_SUM, comm) >> utilities::MpiUtilities::checkError;
    if (globalFailedPoints) {
        PetscMPIInt rank;
        MPI_Comm_rank(comm, &rank) >> utilities::MpiUtilities::checkError;

        std::stringstream failedPointsMessage;
        // march over each failed point
        for (auto& p : failedPoints) {
            // Get the coordinate
            PetscReal centroid[3] = {0.0, 0.0, 0.0};
            DMPlexComputeCellGeometryFVM(GetDM(), p, nullptr, centroid, nullptr) >> utilities::PetscUtilities::checkError;

            failedPointsMessage << "Nan/Inf Point (" << p << ") at [" << centroid[0] << "," << centroid[1] << ", " << centroid[2] << "] on rank " << rank << std::endl;

            // output the labels at this point
            PetscInt numberLabels;
            DMGetNumLabels(GetDM(), &numberLabels);
            failedPointsMessage << "\tLabels: ";
            for (PetscInt l = 0; l < numberLabels; l++) {
                DMLabel labelCheck;
                DMGetLabelByNum(GetDM(), l, &labelCheck) >> utilities::PetscUtilities::checkError;
                const char* labelName;
                DMGetLabelName(GetDM(), l, &labelName) >> utilities::PetscUtilities::checkError;
                PetscInt labelCheckValue;
                DMLabelGetValue(labelCheck, p, &labelCheckValue) >> utilities::PetscUtilities::checkError;
                PetscInt labelDefaultValue;
                DMLabelGetDefaultValue(labelCheck, &labelDefaultValue) >> utilities::PetscUtilities::checkError;
                if (labelDefaultValue != labelCheckValue) {
                    failedPointsMessage << labelName << "(" << labelCheckValue << ") ";
                }
            }
            failedPointsMessage << std::endl;

            // check each local field
            for (const auto& field : GetFields()) {
                {
                    PetscSection section;
                    DMGetGlobalSection(GetDM(), &section) >> utilities::PetscUtilities::checkError;

                    // make sure that this field lives at this point
                    PetscInt dof;
                    PetscSectionGetFieldDof(section, p, field.id, &dof) >> utilities::PetscUtilities::checkError;

                    // get the value at the point
                    const PetscScalar* solutionAtP = nullptr;
                    DMPlexPointGlobalFieldRead(GetDM(), p, field.id, solutionArray, &solutionAtP) >> utilities::PetscUtilities::checkError;
                    if (dof) {
                        failedPointsMessage << '\t' << field.name << ":" << std::endl;
                        for (PetscInt c = 0; c < dof; ++c) {
                            failedPointsMessage << "\t\t[" << c << "]: " << solutionAtP[c] << std::endl;
                        }
                    }
                }

                if (globSourceVector) {
                    PetscSection section;
                    DMGetGlobalSection(sourceDM, &section) >> utilities::PetscUtilities::checkError;

                    // make sure that this field lives at this point
                    PetscInt dof;
                    PetscSectionGetFieldDof(section, p, field.id, &dof) >> utilities::PetscUtilities::checkError;

                    // get the value at the point
                    const PetscScalar* sourceAtP = nullptr;
                    DMPlexPointGlobalFieldRead(sourceDM, p, field.id, sourceArray, &sourceAtP) >> utilities::PetscUtilities::checkError;
                    if (dof) {
                        failedPointsMessage << '\t' << field.name << " source:" << std::endl;
                        for (PetscInt c = 0; c < dof; ++c) {
                            failedPointsMessage << "\t\t[" << c << "]: " << sourceAtP[c] << std::endl;
                        }
                    }
                }
            }
        }

        PetscSynchronizedPrintf(comm, "%s", failedPointsMessage.str().c_str()) >> utilities::PetscUtilities::checkError;
        PetscSynchronizedFlush(comm, PETSC_STDOUT) >> utilities::PetscUtilities::checkError;
    }

    // cleanup
    VecRestoreArrayRead(solutionVec, &solutionArray) >> utilities::PetscUtilities::checkError;
    if (globSourceVector) {
        VecRestoreArrayRead(globSourceVector, &sourceArray) >> utilities::PetscUtilities::checkError;
    }

    return (bool)globalFailedPoints;
}
