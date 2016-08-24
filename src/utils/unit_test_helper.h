#ifndef HELPERS_H
#define HELPERS_H

#include "gtest/gtest.h"
#include "gmock/gmock.h"

MATCHER_P(FloatNearPointwise, tol, "Out of range") {
    return (std::get<0>(arg)>std::get<1>(arg)-tol && std::get<0>(arg)<std::get<1>(arg)+tol);
}

#endif
