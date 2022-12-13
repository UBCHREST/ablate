#ifndef ABLATELIBRARY_LOG_HPP
#define ABLATELIBRARY_LOG_HPP

#include <petsc.h>
#include <memory>
#include <ostream>
#include <vector>

namespace ablate::monitors::logs {
class Log {
   private:
    bool initialized = false;

    /**
     * Private class to wrap the default Log in to a stream buf
     */
    class DefaultOutBuffer : public std::streambuf {
       private:
        Log& log;

       protected:
        int_type overflow(int_type c) override {
            if (c != EOF) {
                log.Printf("%c", static_cast<char>(c));
            }
            return c;
        }

       public:
        DefaultOutBuffer(Log& log) : log(log) {}
    };

    // store pointer for an ostream
    std::unique_ptr<std::ostream> ostream;

    // store the pointer for a stream buffer
    std::unique_ptr<DefaultOutBuffer> ostreambuf;

   public:
    virtual ~Log();

    // each log must support the print command
    virtual void Printf(const char*, ...) = 0;
    virtual void Print(const char* value) { Printf(value); }
    virtual void Print(const char* name, std::size_t num, const double*, const char* format = nullptr);
    virtual void Initialize(MPI_Comm comm = MPI_COMM_SELF) { initialized = true; }

    // built in support calls
    void Print(const char* name, const std::vector<double>& values, const char* format = nullptr);

    // determine if the log has been initialized
    inline const bool& Initialized() const { return initialized; }

    /**
     * Return access to an underlying stream.  Default implementation creates a stream that calls print
     * @return
     */
    virtual std::ostream& GetStream();
};
}  // namespace ablate::monitors::logs

#endif  // ABLATELIBRARY_LOG_HPP
