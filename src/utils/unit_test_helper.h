#ifndef HELPERS_H
#define HELPERS_H

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <complex>
#include <string>

MATCHER_P(FloatNearPointwise, tol, "Out of range") {
    return (std::get<0>(arg) > std::get<1>(arg)-tol && std::get<0>(arg) < std::get<1>(arg)+tol);
}

MATCHER(ComplexConjugate, "z2 is not the complex conjugate of z1") {
  auto z1 = std::get<0>(arg);
  auto z2 = std::get<1>(arg);
  return z1 == std::conj(z2);
}

MATCHER_P2(IsBetween, a, b, std::string(negation ? "isn't" : "is") + " between " + testing::PrintToString(a) + " and " + testing::PrintToString(b)) {
  return a <= arg && arg <= b;
}

#endif
