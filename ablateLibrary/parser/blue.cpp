//
// Created by Matt McGurn on 2/18/21.
//

#include "blue.hpp"
#include <iostream>

#include "registrar.hpp"
static bool item_s_registered = ablate::parser::Registrar<ITalky>::Register<Blue>(
    "Blue",
    "all about blue",
    ablate::parser::ArgumentIdentifier<std::string>{.inputName="whatToSay"},
    ablate::parser::ArgumentIdentifier<ITalky>{.inputName="talky"});

//REGISTER(ITalky, Blue);

void Blue::Talk() {
    std::cout << "I'm blue talking here" << content << std::endl;
}
