#ifndef ABLATELIBRARY_TAGLABELBOUNDARY_HPP
#define ABLATELIBRARY_TAGLABELBOUNDARY_HPP

#include "modifier.hpp"

namespace ablate::domain::modifiers {

/**
 * Class to label/tag all faces on a label boundary
 */
class TagLabelBoundary : public Modifier {
   private:
    // the label to tag
    const std::string name;

    // the label to tag the boundary
    const std::string boundaryName;

    // value of the label to tag the boundary of
    const PetscInt labelValue;

    // value of the boundary label
    const PetscInt boundaryLabelValue;

   public:
    explicit TagLabelBoundary(std::string labelName, std::string boundaryName, int labelValue = {}, int boundaryLabelValue = {});

    void Modify(DM&) override;
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_TAGLABELBOUNDARY_HPP
