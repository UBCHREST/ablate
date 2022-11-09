#ifndef ABLATELIBRARY_TRANSLATEMODIFER_HPP
#define ABLATELIBRARY_TRANSLATEMODIFER_HPP

#include "meshMapper.hpp"
#include "modifier.hpp"
#include <array>

namespace ablate::domain::modifiers {

class Translate : public MeshMapper {
   private:
    PetscReal translate[3] = {0.0, 0.0, 0.0};

   public:
    /**
     * Translates the mesh by x, y, z
     */
    explicit Translate(std::vector<double> translate);

    /**
     * Provide name of modifier for debug/output
     * @return
     */
    std::string ToString() const override;

   private:
    static PetscErrorCode TranslateFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_MESHMAPPER_HPP
