#ifndef ABLATECLIENTTEMPLATE_CONVERGENCETESTER_HPP
#define ABLATECLIENTTEMPLATE_CONVERGENCETESTER_HPP

#include <petsc.h>
#include <memory>
#include <monitors/logs/log.hpp>
#include <vector>

namespace testingResources {

class ConvergenceTester {
   private:
    std::vector<PetscReal> hHistory;
    std::vector<std::vector<PetscReal>> errorHistory;
    const std::string name;
    const std::shared_ptr<ablate::monitors::logs::Log> log;

   public:
    ConvergenceTester(std::string name, std::shared_ptr<ablate::monitors::logs::Log> = {});

    void Record(PetscReal h, const std::vector<PetscReal>& error);

    bool CompareConvergenceRate(const std::vector<PetscReal>& expectedConvergenceRate, std::string& message);
};

}  // namespace testingResources
#endif  // ABLATECLIENTTEMPLATE_CONVERGENCETESTER_HPP
