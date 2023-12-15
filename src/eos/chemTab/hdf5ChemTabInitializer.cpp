#include "hdf5ChemTabInitializer.hpp"

#include <utility>
#include "domain/domain.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::eos::chemTab::Hdf5ChemTabInitializer::Hdf5ChemTabInitializer(std::filesystem::path hdf5Path, std::shared_ptr<ablate::eos::EOS> chemTab, std::shared_ptr<ablate::domain::Region> region)
    : Hdf5Initializer(std::move(hdf5Path), std::move(region)), chemTab(std::dynamic_pointer_cast<eos::ChemTab>(chemTab)) {
    if (!chemTab) {
        throw std::invalid_argument("The equation of state used for Hdf5ChemTabInitializer must be ChemTab");
    }
}

std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> ablate::eos::chemTab::Hdf5ChemTabInitializer::GetFieldFunctions(const std::vector<domain::Field>& fields) const {
    // Create a mesh that the field functions will share
    auto baseMesh = std::make_shared<Hdf5Mesh>(hdf5Path);

    std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> functions;

    // March over each requested field
    for (const auto& field : fields) {
        // get the math function
        std::shared_ptr<ablate::mathFunctions::MathFunction> mathFunction;

        // check to see if this is the progress field
        if (field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD) {
            // use the special mapping function
            mathFunction = std::make_shared<Hdf5ChemTabMappingMathFunction>(baseMesh, *chemTab);
        } else if (field.name == ablate::eos::ChemTab::DENSITY_YI_DECODE_FIELD) {
            // just skip the init
            continue;
        } else {
            // get use the field directly
            mathFunction = std::make_shared<Hdf5MathFunction>(baseMesh, ablate::domain::Domain::solution_vector_name + "_" + field.name);
        }

        // Create a fieldFunction wrapper
        auto fieldFunction = std::make_shared<ablate::mathFunctions::FieldFunction>(field.name, mathFunction, nullptr, region);

        functions.push_back(fieldFunction);
    }
    return functions;
}

ablate::eos::chemTab::Hdf5ChemTabInitializer::Hdf5ChemTabMappingMathFunction::Hdf5ChemTabMappingMathFunction(const std::shared_ptr<Hdf5Mesh>& baseMesh, const ablate::eos::ChemTab& chemTab)
    : Hdf5MathFunction(baseMesh, ablate::domain::Domain::solution_vector_name + "_" + ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD),
      chemTab(chemTab),
      eulerFunction(std::make_shared<Hdf5MathFunction>(baseMesh, ablate::domain::Domain::solution_vector_name + "_" + ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD)),
      numberOfSpecies((PetscInt)chemTab.GetSpeciesNames().size()) {
    // resize the number of based off the new progress variables
    resultSize = (PetscInt)chemTab.GetProgressVariables().size();
}

PetscErrorCode ablate::eos::chemTab::Hdf5ChemTabInitializer::Hdf5ChemTabMappingMathFunction::Eval(PetscInt xyzDim, const PetscReal xyz[], PetscScalar* u) const {
    PetscFunctionBeginUser;

    // Size up a scratch array to hold the number of yi's components. Note at this time yi is yi*density
    std::vector<PetscReal> yis(components);
    PetscCall(ablate::domain::Hdf5Initializer::Hdf5MathFunction::Eval(xyzDim, xyz, yis.data()));

    // make sure the number of components is equal
    if (numberOfSpecies != (PetscInt)chemTab.GetSpeciesNames().size()) {
        throw std::invalid_argument("There appears to be a missmatch between the chemTabModel and species used in the hdf5 file.");
    }

    // compute the density here.  This assumes that
    PetscReal euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + xyzDim];  // max size of euler
    PetscCall(eulerFunction->Eval(xyzDim, xyz, euler));

    // convert densityYi to yi
    for (auto& yi : yis) {
        yi /= euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
    }

    // Map to the progress variables
    std::vector<PetscReal> progressVariables(chemTab.GetProgressVariables().size());
    chemTab.ComputeProgressVariables(yis, progressVariables);

    // scale and put back the progress variables into u
    for (std::size_t i = 0; i < progressVariables.size(); ++i) {
        u[i] = progressVariables[i] * euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::domain::Initializer, ablate::eos::chemTab::Hdf5ChemTabInitializer, "Initializes the domain (assuming ChemTab) using a previous result stored in an hdf5 file.",
         ARG(std::filesystem::path, "path", "path to hdf5 file"), ARG(ablate::eos::EOS, "chemTab", "the chemTab EOS"));
