#ifndef ABLATELIBRARY_STREAMLOG_HPP
#define ABLATELIBRARY_STREAMLOG_HPP

#include <iostream>
#include <vector>
#include "log.hpp"

namespace ablate::monitors::logs {
class StreamLog : public Log {
   private:
    std::ostream& stream;
    std::vector<char> buffer;

   public:
    explicit StreamLog(std::ostream& stream = std::cout);

    void Print(const char*) override;
    void Printf(const char*, ...) override;

    void Initialize(MPI_Comm comm) override;

    /**
     * Return access to an underlying stream
     * @return
     */
    std::ostream& GetStream() override { return stream; }
};
}  // namespace ablate::monitors::logs

#endif  // ABLATELIBRARY_STREAMLOG_HPP
