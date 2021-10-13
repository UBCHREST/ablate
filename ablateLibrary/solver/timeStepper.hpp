#ifndef ABLATELIBRARY_TIMESTEPPER_H
#define ABLATELIBRARY_TIMESTEPPER_H
#include <petsc.h>
#include <io/serializer.hpp>
#include <map>
#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "monitors/monitor.hpp"
#include "solvable.hpp"
#include "utilities/loggable.hpp"
#include <functional>

namespace ablate::solver {
class TimeStepper : public std::enable_shared_from_this<TimeStepper>, private utilities::Loggable<TimeStepper> {
   private:
    TS ts;                                                    /** The PETSC time stepper**/
    std::string name;                                         /** the name for this time stepper **/
    std::vector<std::shared_ptr<monitors::Monitor>> monitors; /** the monitors **/

    // Store a pointer to the Serializer
    const std::shared_ptr<io::Serializer> serializer;

    // Static calls to be passed to the Petsc TS
    static PetscErrorCode TSPreStageFunction(TS ts, PetscReal stagetime);
    static PetscErrorCode TSPreStepFunction(TS ts);
    static PetscErrorCode TSPostStepFunction(TS ts);
    static PetscErrorCode TSPostEvaluateFunction(TS ts);

    // pre and post step functions for the flow
    std::vector<std::function<void(TS ts)>> preStepFunctions;
    std::vector<std::function<void(TS ts, PetscReal)>> preStageFunctions;
    std::vector<std::function<void(TS ts)>> postStepFunctions;
    std::vector<std::function<void(TS ts)>> postEvaluateFunctions;

   public:
    TimeStepper(std::string name, std::map<std::string, std::string> arguments, std::shared_ptr<io::Serializer> serializer = {});
    ~TimeStepper();

    TS& GetTS() { return ts; }

    void Solve(std::shared_ptr<Solvable>);

    void AddMonitor(std::shared_ptr<monitors::Monitor>);

    void Register(std::weak_ptr<io::Serializable> serializable);

    double GetTime() const;

    const std::string& GetName() const { return name; }


    /**
     * Adds function to be called before each flow step
     * @param preStep
     */
    void RegisterPreStep(std::function<void(TS ts)> preStep) { this->preStepFunctions.push_back(preStep); }

    /**
     * Adds function to be called before each flow step
     * @param preStep
     */
    void RegisterPreStage(std::function<void(TS ts, PetscReal)> preStage) { this->preStageFunctions.push_back(preStage); }

    /**
     * Adds function to be called after each flow step
     * @param preStep
     */
    void RegisterPostStep(std::function<void(TS ts)> postStep) { this->postStepFunctions.push_back(postStep); }

    /**
     * Adds function after each evaluated.  This is where the solution can be modified if needed.
     * @param postStep
     */
    void RegisterPostEvaluate(std::function<void(TS ts)> postEval) { this->postEvaluateFunctions.push_back(postEval); }


};
}  // namespace ablate::solve

#endif  // ABLATELIBRARY_TIMESTEPPER_H
