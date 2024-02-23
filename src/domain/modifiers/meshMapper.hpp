#ifndef ABLATELIBRARY_MESHMAPPER_HPP
#define ABLATELIBRARY_MESHMAPPER_HPP

#include <memory>
#include "domain/region.hpp"
#include "mathFunctions/mathFunction.hpp"
#include "modifier.hpp"

namespace ablate::domain::modifiers {

class MeshMapper : public Modifier {
   private:
    //! The mapping function that takes input coordinates and maps them to output
    const std::shared_ptr<ablate::mathFunctions::MathFunction> mappingFunction;

    //! an optional region to apply the mesh mapping
    const std::shared_ptr<ablate::domain::Region> mappingRegion{};

   public:
    /**
     * General constructor for all mesh mappers
     * @param mappingFunction The mapping function that takes input coordinates and maps them to output
     * @param mappingRegion an optional region to apply the mesh mapping
     */
    explicit MeshMapper(std::shared_ptr<ablate::mathFunctions::MathFunction> mappingFunction, std::shared_ptr<ablate::domain::Region> mappingRegion = {});

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
     * Provide name of modifier for debug/output
     * @return
     */
    [[nodiscard]] std::string ToString() const override { return "ablate::domain::modifiers::MeshMapper"; }
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_MESHMAPPER_HPP
