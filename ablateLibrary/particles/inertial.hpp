#ifndef ABLATELIBRARY_INERTIAL_HPP
#define ABLATELIBRARY_INERTIAL_HPP
#include "particles.hpp"

namespace ablate::particles {

class Inertial : public Particles {
   private:
    //  fluid density needed for particle drag force
    PetscReal fluidDensity;
    // fluid viscosity needed for particle drag force
    PetscReal fluidViscosity;
    // gravity field
    PetscReal gravityField[3] = {0, 0, 0};

    /*
     * Kinematics vector is a combination of particle velocity and position
     * in order to pass into TSSolve.
     */
    static PetscErrorCode PackKinematics(TS ts, Vec position, Vec velocity, Vec kinematics);

    /*
     * Unpack kinematics to get particle position and velocity
     */
    static PetscErrorCode UnpackKinematics(TS ts, Vec kinematics, Vec position, Vec velocity);

    /* calculating RHS of the following equations
     * x_t = vp
     * u_t = f(vf-vp)/tau_p + g(1-\rho_f/\rho_p)
     */
    static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx);

    void advectParticles(TS flowTS);

   public:
    Inertial(std::string name, int ndims, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<particles::initializers::Initializer> initializer, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> fieldInitialization,
             std::shared_ptr<mathFunctions::MathFunction> exactSolution = {}, std::shared_ptr<parameters::Parameters> options = {});
    ~Inertial() override;

    void InitializeFlow(std::shared_ptr<flow::Flow> flow) override;

    inline static const char FluidVelocity[] = "FluidVelocity";
    inline static const char ParticleKinematics[] = "ParticleKinematics";
};

}  // namespace ablate::particles

#endif  // ABLATELIBRARY_INERTIAL_HPP
