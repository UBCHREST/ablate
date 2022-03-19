//
// Created by owen on 3/19/22.
//

#ifndef ABLATELIBRARY_RADIATIONPROCESS_H
#define ABLATELIBRARY_RADIATIONPROCESS_H

namespace ablate::boundarySolver {

class RadiationProcess {
   public:
    virtual ~RadiationProcess() = default;
    virtual void Initialize(ablate::radiationSolver::RadiationSolver& bSolver) = 0;
};

}  // namespace ablate::radiationSolver

#endif  // ABLATELIBRARY_RADIATIONPROCESS_H
