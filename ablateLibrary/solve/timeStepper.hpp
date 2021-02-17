
#ifndef ABLATELIBRARY_TIMESTEPPER_H
#define ABLATELIBRARY_TIMESTEPPER_H
#include <petsc.h>
#include <map>
#include "solvable.hpp"
#include <memory>

namespace ablate {
namespace solve {
class TimeStepper {
   private:
    TS ts;            /** The PETSC time stepper**/
    MPI_Comm comm;    /** the comm used for this ts and any children**/
    std::string name; /** the name for this time stepper **/

   public:
    TimeStepper(MPI_Comm comm, std::string name, std::map<std::string, std::string> arguments);
    ~TimeStepper();

    const TS& GetTS() const { return ts; }

    void Solve(std::shared_ptr<Solvable>);
};
}
}

#endif  // ABLATELIBRARY_TIMESTEPPER_H
