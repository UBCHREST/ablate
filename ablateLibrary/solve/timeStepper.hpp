#ifndef ABLATELIBRARY_TIMESTEPPER_H
#define ABLATELIBRARY_TIMESTEPPER_H
#include <petsc.h>
#include <map>
#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "monitors/monitor.hpp"
#include "solvable.hpp"

namespace ablate::solve {
class TimeStepper {
   private:
    TS ts;                                                    /** The PETSC time stepper**/
    std::string name;                                         /** the name for this time stepper **/
    std::vector<std::shared_ptr<monitors::Monitor>> monitors; /** the monitors **/

    // Store a petsc class id used for flow used for logging
    inline static PetscClassId petscClassId = 0;

    PetscLogEvent tsLogEvent;

   public:
    TimeStepper(std::string name, std::map<std::string, std::string> arguments);
    ~TimeStepper();

    TS& GetTS() { return ts; }

    void Solve(std::shared_ptr<Solvable>, std::shared_ptr<parameters::Parameters> restartParameters = nullptr);

    void AddMonitor(std::shared_ptr<monitors::Monitor>);

    double GetTime() const;

    static PetscClassId GetPetscClassId();
};
}  // namespace ablate::solve

#endif  // ABLATELIBRARY_TIMESTEPPER_H
