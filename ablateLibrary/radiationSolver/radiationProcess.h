//
// Created by owen on 3/19/22.
//

#ifndef ABLATELIBRARY_RADIATIONPROCESS_H
#define ABLATELIBRARY_RADIATIONPROCESS_H

#include "radiationSolver.h"

namespace ablate::radiationSolver {

class RadiationProcess {
   public:
    virtual ~RadiationProcess() = default;
    virtual void Initialize(ablate::radiationSolver::RadiationSolver& bSolver) = 0;
};

}  // namespace ablate::radiationSolver

#endif  // ABLATELIBRARY_RADIATIONPROCESS_H
