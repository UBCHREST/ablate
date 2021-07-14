#ifndef ABLATELIBRARY_STREAMLOG_HPP
#define ABLATELIBRARY_STREAMLOG_HPP

#include "log.hpp"
#include <iostream>
#include <vector>

namespace ablate::monitors::logs {
    class StreamLog : public Log {
       private:
        std::ostream& stream;
        bool output = true;
        std::vector<char> buffer;

       public:
        explicit StreamLog(std::ostream& stream = std::cout);

        void Print(const char*) override;
        void Printf(const char*, ...) override;

        void Initialize(MPI_Comm comm) override;

    };
}

#endif  // ABLATELIBRARY_STREAMLOG_HPP
