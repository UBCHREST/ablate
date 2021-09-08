#ifndef ABLATELIBRARY_TIMESTEPPER_H
#define ABLATELIBRARY_TIMESTEPPER_H
#include <petsc.h>
#include <environment/restartManager.hpp>
#include <map>
#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "monitors/monitor.hpp"
#include "solvable.hpp"
#include "environment/restartable.hpp"

namespace ablate::solve {
class TimeStepper: public std::enable_shared_from_this<TimeStepper>, public ablate::environment::Restartable {
   private:
    TS ts;                                                    /** The PETSC time stepper**/
    std::string name;                                         /** the name for this time stepper **/
    std::vector<std::shared_ptr<monitors::Monitor>> monitors; /** the monitors **/

    // Store a petsc class id used for flow used for logging
    inline static PetscClassId petscClassId = 0;

    PetscLogEvent tsLogEvent;

    // Store a pointer to the restartManager
    std::shared_ptr<ablate::environment::RestartManager> restartManager;

   public:
    TimeStepper(std::string name, std::map<std::string, std::string> arguments);
    ~TimeStepper() override;

    TS& GetTS() { return ts; }

    void Solve(std::shared_ptr<Solvable>, std::shared_ptr<ablate::environment::RestartManager> restartManager = {});

    void AddMonitor(std::shared_ptr<monitors::Monitor>);

    double GetTime() const;

    static PetscClassId GetPetscClassId();

    const std::string& GetName() const override{
        return name;
    }

    void Save(environment::SaveState&) const override;

    void Restore(const environment::RestoreState&) override;
};
}  // namespace ablate::solve

#endif  // ABLATELIBRARY_TIMESTEPPER_H
