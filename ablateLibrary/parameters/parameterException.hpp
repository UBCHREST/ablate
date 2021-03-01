#ifndef ABLATELIBRARY_PARAMETEREXCEPTION_HPP
#define ABLATELIBRARY_PARAMETEREXCEPTION_HPP

struct ParameterException : public std::exception {
   private:
    std::string message;

   public:
    ParameterException(std::string variableName) { message = "The variable " + variableName + " cannot be found in the parameters."; }

    const char* what() const throw() { return message.c_str(); }
};
#endif  // ABLATELIBRARY_PARAMETEREXCEPTION_HPP
