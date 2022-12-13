#ifndef ABLATELIBRARY_LES_H
#define ABLATELIBRARY_LES_H

#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "navierStokesTransport.hpp"

namespace ablate::finiteVolume::processes {

class LES : public FlowProcess {
   private:
    /**
     * tke is assumed to consist of a field with a single component
     */
    const std::string tkeField;

    // constant values
    inline const static PetscReal c_k = 0.094;
    inline const static PetscReal c_e = 1.048;
    inline const static PetscReal c_p = 1004.0;
    inline const static PetscReal scT = 1.00;
    inline const static PetscReal prT = 1.00;

    // Store the numberComponents for each ev/species
    std::vector<PetscInt> numberComponents;

   public:
    /**
     * The field name containing the tkeField.  This is assumed to contain the density/tke value (conserved form)
     * @param tkeField
     */
    explicit LES(std::string tkeField = {});

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

   public:
    /**
     * This computes the momentum source for SGS model for rhoU
     * f = "euler"
     * u = {"euler"}
     * a = {"tke, "vel"}
     * ctx = nullptr
     * @return
     */
    static PetscErrorCode LesMomentumFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);
    /**
     * This computes the energy source for SGS model for rhoE
     * f = "euler"
     * u = {"euler"}
     * a = {"tke", "vel", "temperature"}
     * ctx = nullptr
     * @return
     */
    static PetscErrorCode LesEnergyFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * This computes the EV transfer for SGS model for density_tke
     * f = "conserved_tke"
     * u = {"euler"}
     * a = {"tke", "vel"}
     * ctx = nullptr
     * return
     */
    static PetscErrorCode LesTkeFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * This computes the species transfer for SGS model for density-Yi or density ev
     * f = "conserved_ev/yi"
     * u = {"euler"}
     * a = {"tke", "yi/ev"}
     * ctx = (PetscInt*) size of yi/ev field
     * @return
     */
    static PetscErrorCode LesEvFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[],
                                    const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * static support function to compute the turbulent viscosity
     * @param dim
     * @param fg
     * @param field
     * @param uOff
     * @param mut
     * @return
     */
    static PetscErrorCode LesViscosity(PetscInt dim, const PetscFVFaceGeom* fg, const PetscScalar* densityField, const PetscReal turbulence, PetscReal& mut);
};
}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_LES_H
