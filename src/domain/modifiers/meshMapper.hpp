#ifndef ABLATELIBRARY_MESHMAPPER_HPP
#define ABLATELIBRARY_MESHMAPPER_HPP

#include <memory>
#include "mathFunctions/mathFunction.hpp"
#include "modifier.hpp"

namespace ablate::domain::modifiers {

class MeshMapper : public Modifier {
   private:
    const std::shared_ptr<ablate::mathFunctions::MathFunction> mappingFunction;

   public:
    /**
     * General constructor for all mesh mappers
     */
    explicit MeshMapper(std::shared_ptr<ablate::mathFunctions::MathFunction>);

    /**
     * March over each vertex in the cell and map using the supplied function
     */
    void Modify(DM&) override;

    /**
     * This returns a single modified point value.
     * @param in
     * @param out will be resized to match in
     */
    void Modify(const std::vector<double>& in, std::vector<double>& out) const;

    /**
     * This modifies a single point in memory
     * @param size the size of the vector
     * @param coord the coordinate to modify
     */
    void Modify(PetscInt size, PetscReal* coord) const;

    /**
     * Provide name of modifier for debug/output
     * @return
     */
    [[nodiscard]] std::string ToString() const override { return "ablate::domain::modifiers::MeshMapper"; }
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_MESHMAPPER_HPP
