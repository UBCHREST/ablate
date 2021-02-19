//
// Created by Matt McGurn on 2/18/21.
//

#ifndef FACTORYPLAYGROUND_ITALKY_HPP
#define FACTORYPLAYGROUND_ITALKY_HPP

class ITalky{
public:
    virtual void Talk() =0;
    virtual ~ITalky() = default;
};



#endif //FACTORYPLAYGROUND_ITALKY_HPP
