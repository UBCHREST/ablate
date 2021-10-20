#ifndef ABLATELIBRARY_SOLVER_HPP
#define ABLATELIBRARY_SOLVER_HPP

#include <domain/region.hpp>
#include <domain/subDomain.hpp>
#include <parameters/parameters.hpp>
#include <string>
#include <vector>
#include "io/serializable.hpp"

namespace ablate::solver {

class TimeStepper;

class Solver : public io::Serializable {
   private:
    // pre and post step functions for the flow
    std::vector<std::function<void(TS ts, Solver&)>> preStepFunctions;
    std::vector<std::function<void(TS ts, Solver&, PetscReal)>> preStageFunctions;
    std::vector<std::function<void(TS ts, Solver&)>> postStepFunctions;
    std::vector<std::function<void(TS ts, Solver&)>> postEvaluateFunctions;

    // The name of this solver
    const std::string solverId;

    // The region of this solver.
    const std::shared_ptr<domain::Region> region;

   protected:
    // an optional petscOptions that is used for this solver
    PetscOptions petscOptions;

    // use the subDomain to setup the problem
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    // The constructor to be call by any Solve implementation
    explicit Solver(std::string solverId, std::shared_ptr<domain::Region> = {}, std::shared_ptr<parameters::Parameters> options = nullptr);

    // function to decompress FieldDescriptors
    void DecompressFieldFieldDescriptor(std::vector<ablate::domain::FieldDescriptor>& FieldDescriptors);

   public:
    virtual ~Solver();

    /** Register all needed fields with the subDomain **/
    virtual void Register(std::shared_ptr<ablate::domain::SubDomain> subDomain);

    /** Setup and size the subDomain with the subDomain **/
    virtual void Setup() = 0;

    /** Finalize the Setup of the subDomain before running **/
    virtual void Initialize() = 0;

    inline ablate::domain::SubDomain& GetSubDomain() { return *subDomain; }

    inline std::shared_ptr<domain::Region> GetRegion() const { return region; }

    inline const std::string& GetId() const override { return solverId; }

    // Support for timestepping calls
    void PreStage(TS ts, PetscReal stagetime);
    void PreStep(TS ts);
    void PostStep(TS ts);
    void PostEvaluate(TS ts);

    /**
     * Adds function to be called before each flow step
     * @param preStep
     */
    inline void RegisterPreStep(std::function<void(TS ts, Solver&)> preStep) { this->preStepFunctions.push_back(preStep); }

    /**
     * Adds function to be called before each flow step
     * @param preStep
     */
    inline void RegisterPreStage(std::function<void(TS ts, Solver&, PetscReal)> preStage) { this->preStageFunctions.push_back(preStage); }

    /**
     * Adds function to be called after each flow step
     * @param preStep
     */
    inline void RegisterPostStep(std::function<void(TS ts, Solver&)> postStep) { this->postStepFunctions.push_back(postStep); }

    /**
     * Adds function after each evaluated.  This is where the solution can be modified if needed.
     * @param postStep
     */
    inline void RegisterPostEvaluate(std::function<void(TS ts, Solver&)> postEval) { this->postEvaluateFunctions.push_back(postEval); }

    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) const override;
    void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;
};

}  // namespace ablate::solver
#endif  // ABLATELIBRARY_SOLVER_HPP
