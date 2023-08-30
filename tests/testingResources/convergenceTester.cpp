#include "convergenceTester.hpp"
#include <monitors/logs/stdOut.hpp>
#include <utility>
#include "PetscTestErrorChecker.hpp"

testingResources::ConvergenceTester::ConvergenceTester(std::string name, const std::shared_ptr<ablate::monitors::logs::Log>& logIn)
    : name(std::move(name)), log(logIn ? logIn : std::make_shared<ablate::monitors::logs::StdOut>()) {}
void testingResources::ConvergenceTester::Record(PetscReal h, const std::vector<PetscReal>& error) {
    hHistory.push_back(PetscLog10Real(h));

    if (error.size() != errorHistory.size()) {
        errorHistory.resize(error.size());
    }

    for (std::size_t b = 0; b < errorHistory.size(); b++) {
        errorHistory[b].push_back(PetscLog10Real(error[b]));
    }

    if (log) {
        auto errorName = name + "(" + std::to_string(hHistory.size()) + ")";
        log->Print(errorName.c_str(), error, "%g");
        log->Print("\n");
    }
}

bool testingResources::ConvergenceTester::CompareConvergenceRate(const std::vector<PetscReal>& expectedConvergenceRate, std::string& message, bool checkDifference) {
    PetscTestErrorChecker testErrorChecker;
    bool passed = true;

    if (expectedConvergenceRate.size() != errorHistory.size()) {
        throw std::invalid_argument("The expectedConvergenceRate (" + std::to_string(expectedConvergenceRate.size()) + ")is of incorrect size for the error (" + std::to_string(errorHistory.size()) +
                                    ")");
    }

    for (std::size_t b = 0; b < errorHistory.size(); b++) {
        PetscReal slope;
        PetscReal intercept;
        PetscLinearRegression((PetscReal)hHistory.size(), &hHistory[0], &errorHistory[b][0], &slope, &intercept) >> testErrorChecker;

        if (log) {
            log->Printf("%s Convergence[%d]: %g\n", name.c_str(), b, slope);
        }

        if (std::isnan(expectedConvergenceRate[b]) != std::isnan(slope)) {
            passed = false;
            message += "incorrect L2 convergence order for component[" + std::to_string(b) + "] ";
        } else if (checkDifference) {
            if (PetscAbs(slope - expectedConvergenceRate[b]) > 0.2) {
                passed = false;
                message += "incorrect L2 convergence order for component[" + std::to_string(b) + "]. Value must be near " + std::to_string(expectedConvergenceRate[b]);
            }
        } else {
            if (slope < expectedConvergenceRate[b]) {
                passed = false;
                message += "incorrect L2 convergence order for component[" + std::to_string(b) + "].  Value must be greater than/equal to" + std::to_string(expectedConvergenceRate[b]);
            }
        }
    }
    return passed;
}
