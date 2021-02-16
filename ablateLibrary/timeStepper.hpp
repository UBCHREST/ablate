
#ifndef ABLATELIBRARY_TIMESTEPPER_H
#define ABLATELIBRARY_TIMESTEPPER_H
#include <petsc.h>
#include <map>

namespace ablate {
class TimeStepper {
   private:
    TS ts; /** The PETSC time stepper**/
    MPI_Comm comm; /** the comm used for this ts and any children**/
    std::string name; /** the name for this time stepper **/

   public:
    TimeStepper(MPI_Comm comm, std::string name, std::map<std::string, std::string> arguments);
    ~TimeStepper();
};
}

#endif  // ABLATELIBRARY_TIMESTEPPER_H
