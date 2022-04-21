#ifndef ABLATELIBRARY_STDOUT_HPP
#define ABLATELIBRARY_STDOUT_HPP
#include "log.hpp"

namespace ablate::monitors::logs {
class StdOut : public Log {
   private:
    bool output = true;

   public:
    void Printf(const char*, ...) final;

    void Initialize(MPI_Comm comm) final;
};
}  // namespace ablate::monitors::logs

#endif  // ABLATELIBRARY_STDOUT_HPP
