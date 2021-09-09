#ifndef ABLATELIBRARY_TIMESTEPPER_H
#define ABLATELIBRARY_TIMESTEPPER_H
#include <petsc.h>
#include <environment/restartManager.hpp>
#include <io/serializer.hpp>
#include <map>
#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "environment/restartable.hpp"
#include "monitors/monitor.hpp"
#include "solvable.hpp"

namespace ablate::solve {
class TimeStepper: public std::enable_shared_from_this<TimeStepper>{
   private:
    TS ts;                                                    /** The PETSC time stepper**/
    std::string name;                                         /** the name for this time stepper **/
    std::vector<std::shared_ptr<monitors::Monitor>> monitors; /** the monitors **/

    // Store a petsc class id used for flow used for logging
    inline static PetscClassId petscClassId = 0;

    PetscLogEvent tsLogEvent;

    // Store a pointer to the Serializer
    const std::shared_ptr<io::Serializer> serializer;

   public:
    TimeStepper(std::string name, std::map<std::string, std::string> arguments, std::shared_ptr<io::Serializer> serializer = {});
    ~TimeStepper();

    TS& GetTS() { return ts; }

    void Solve(std::shared_ptr<Solvable>);

    void AddMonitor(std::shared_ptr<monitors::Monitor>);

    void Register(std::weak_ptr<io::Serializable> serializable);

    double GetTime() const;

    static PetscClassId GetPetscClassId();

    const std::string& GetName() const{
        return name;
    }

};
}  // namespace ablate::solve

#endif  // ABLATELIBRARY_TIMESTEPPER_H
