#ifndef ABLATECLIENTTEMPLATE_INTERRORCHECKER_HPP
#define ABLATECLIENTTEMPLATE_INTERRORCHECKER_HPP
namespace ablate {
namespace utilities {

class IntErrorChecker {
   private:
    const std::string errorMessage;

   public:
    IntErrorChecker(std::string errorMessage) : errorMessage(errorMessage) {}

    friend void operator>>(int ierr, const IntErrorChecker &errorChecker) {
        if (ierr != 0) {
            throw std::runtime_error(errorChecker.errorMessage + std::to_string(ierr));
        }
    }
};
}  // namespace utilities

}  // namespace ablate

#endif  // ABLATECLIENTTEMPLATE_INTERRORCHECKER_HPP
