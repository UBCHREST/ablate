#ifndef ABLATELIBRARY_STDOUT_HPP
#define ABLATELIBRARY_STDOUT_HPP
#include <iostream>
#include "log.hpp"

namespace ablate::monitors::logs {
class StdOut : public Log {
   private:
    bool output = true;

   public:
    // allow access to all print from base
    using Log::Print;
    void Printf(const char*, ...) final;

    void Initialize(MPI_Comm comm) final;

    /**
     * Return access to an underlying stream
     * @return
     */
    std::ostream& GetStream() override;
};
}  // namespace ablate::monitors::logs

#endif  // ABLATELIBRARY_STDOUT_HPP
