#ifndef ABLATELIBRARY_NULLLOG_HPP
#define ABLATELIBRARY_NULLLOG_HPP

#include <iostream>
#include <ostream>
#include <vector>
#include "log.hpp"

namespace ablate::monitors::logs {
class NullLog : public Log {
   private:
    /**
     * Simple null buffer
     */
    class NullBuffer : public std::streambuf {
       protected:
        int_type overflow(int_type c) override { return c; }
    };

    inline static NullBuffer nullBuffer;

   public:
    /**
     * public static null stream that can be used by others
     */
    inline static std::ostream nullStream = std::ostream(&nullBuffer);

   public:
    // allow access to all print from base
    using Log::Print;
    void Print(const char*) override {}
    void Printf(const char*, ...) override {}
    void Initialize(MPI_Comm comm) override {}

    /**
     * Return access a null stream
     * @return
     */
    std::ostream& GetStream() override { return nullStream; }
};
}  // namespace ablate::monitors::logs

#endif  // ABLATELIBRARY_NULLLOG_HPP
