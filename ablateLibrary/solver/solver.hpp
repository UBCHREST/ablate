#ifndef ABLATELIBRARY_SOLVER_HPP
#define ABLATELIBRARY_SOLVER_HPP

#include <petscoptions.h>
#include <domain/subDomain.hpp>
#include <parameters/parameters.hpp>
#include <string>
#include <vector>
#include "timeStepper.hpp"

namespace ablate::solver {

class Solver {
   protected:
    // The name of this domain.  This will be used for the subdomain
    const std::string name;

    // an optional petscOptions that is used for this solver
    PetscOptions petscOptions;

    // use the subDomain to setup the problem
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    // The constructor to be call by any Solve implementation
    explicit Solver(std::string name, std::shared_ptr<parameters::Parameters> options = nullptr);

   public:
    virtual ~Solver();

    virtual void SetupDomain(std::shared_ptr<ablate::domain::SubDomain> subDomain);

    virtual void CompleteSetup(solver::TimeStepper&) = 0;



};

}
#endif  // ABLATELIBRARY_SOLVER_HPP
