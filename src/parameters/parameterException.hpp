#ifndef ABLATELIBRARY_PARAMETEREXCEPTION_HPP
#define ABLATELIBRARY_PARAMETEREXCEPTION_HPP

namespace ablate::parameters {

struct ParameterException : public std::exception {
   private:
    std::string message;

   public:
    ParameterException(std::string variableName) { message = "The variable " + variableName + " cannot be found in the parameters."; }

    const char* what() const throw() override { return message.c_str(); }
};

}  // namespace ablate::parameters
#endif  // ABLATELIBRARY_PARAMETEREXCEPTION_HPP
