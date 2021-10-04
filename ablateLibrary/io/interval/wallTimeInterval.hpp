#ifndef ABLATELIBRARY_WALLTIMEINTERVAL_HPP
#define ABLATELIBRARY_WALLTIMEINTERVAL_HPP

#include "chrono"
#include "functional"
#include "interval.hpp"
namespace ablate::io::interval {

class WallTimeInterval : public Interval {
   private:
    const std::chrono::seconds timeInterval;
    std::chrono::time_point<std::chrono::system_clock> previousTime;
    const std::function<std::chrono::time_point<std::chrono::system_clock>()> now;

   public:
    explicit WallTimeInterval(int timeInterval, std::function<std::chrono::time_point<std::chrono::system_clock>()> = std::chrono::system_clock::now);
    bool Check(MPI_Comm comm, PetscInt steps, PetscReal time) override;
};

}  // namespace ablate::io::interval

#endif  // ABLATELIBRARY_WALLTIMEINTERVAL_HPP
