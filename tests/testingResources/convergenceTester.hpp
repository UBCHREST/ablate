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
    /**
     * helper class to compare convergence history
     * @param name
     */
    explicit ConvergenceTester(std::string name, const std::shared_ptr<ablate::monitors::logs::Log>& = {});

    void Record(PetscReal h, const std::vector<PetscReal>& error);

    /**
     * perform the compare, output info, and return status
     * @param expectedConvergenceRate
     * @param message
     * @param checkDifference if true checks to see if rates are close, if false, just checks if rate is greater than expected
     * @return true if they match, false if they do not
     */
    bool CompareConvergenceRate(const std::vector<PetscReal>& expectedConvergenceRate, std::string& message, bool checkDifference = true);
};

}  // namespace testingResources
#endif  // ABLATECLIENTTEMPLATE_CONVERGENCETESTER_HPP
