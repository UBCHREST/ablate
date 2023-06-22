#ifndef ABLATELIBRARY_EOS_CHEMTAB_INITIALIZER_HPP
#define ABLATELIBRARY_EOS_CHEMTAB_INITIALIZER_HPP

#include "eos/chemTab.hpp"
#include "mathFunctions/constantValue.hpp"

namespace ablate::eos::chemTab {

/**
 * Class that species progress function from a chemTab model
 */
class Initializer : public ablate::mathFunctions::ConstantValue {
   public:
    /**
     * Determines the progress field for initialization
     * @param initializer
     * @param eos
     */
    explicit Initializer(const std::string& initializer, std::shared_ptr<ablate::eos::EOS> eos);
};

}  // namespace ablate::eos::chemTab
#endif  // ABLATELIBRARY_EOS_CHEMTAB_INITIALIZER_HPP
