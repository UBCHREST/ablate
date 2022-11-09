#ifndef ABLATELIBRARY_MESHMAPPER_HPP
#define ABLATELIBRARY_MESHMAPPER_HPP

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
     * March over each vertex in the cell and mapp using the supplied function
     */
    void Modify(DM&) override;

    /**
     * This returns a single modified point value.
     * @param in
     * @param out will be resized to match in
     */
    void Modify(const std::vector<double>& in, std::vector<double>& out) const;

    /**
     * Provide name of modifier for debug/output
     * @return
     */
    std::string ToString() const override { return "ablate::domain::modifiers::MeshMapper"; }
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_MESHMAPPER_HPP
