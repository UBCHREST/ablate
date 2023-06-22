#include "initializer.hpp"
#include <algorithm>
#include <mathFunctions/functionPointer.hpp>
#include "utilities/vectorUtilities.hpp"

ablate::eos::chemTab::Initializer::Initializer(const std::string &initializer, std::shared_ptr<ablate::eos::EOS> eos)
    : ablate::mathFunctions::ConstantValue([initializer, eos]() {
          auto chemTabModel = std::dynamic_pointer_cast<ablate::eos::ChemTab>(eos);
          if (!chemTabModel) {
              throw std::invalid_argument("The ablate::finiteVolume::fieldFunctions::ChemTabField requires a ablate::eos::ChemTab model");
          }

          // Get the progress variables from ChemTab
          std::vector<double> progressVariables(chemTabModel->GetProgressVariables().size());
          chemTabModel->GetInitializerProgressVariables(initializer, progressVariables);

          return progressVariables;
      }()) {}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::eos::chemTab::Initializer, "Class that species progress variable from a chemTab model",
         ARG(std::string, "initializer", "the name of the initializer in the chemTab model"), ARG(ablate::eos::EOS, "eos", "must be a ablate::eos::ChemTab model"));