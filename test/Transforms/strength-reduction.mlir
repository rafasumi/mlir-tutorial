// RUN: sblp-opt --strength-reduction %s | FileCheck %s

module {
  func.func @power_of_two_rhs(%x: i32) -> i32 {
    %512 = arith.constant 512 : i32
    %result = arith.muli %x, %512 : i32
    return %result : i32
  }
  // CHECK:  func.func @power_of_two_rhs
  // CHECK:    %[[NINE:.*]] = arith.constant 9
  // CHECK:    arith.shli %{{[0-9a-z]+}}, %[[NINE]]

  func.func @power_of_two_lhs(%x: i32) -> i32 {
    %1024 = arith.constant 1024 : i32
    %result = arith.muli %1024, %x : i32
    return %result : i32
  }
  // CHECK:  func.func @power_of_two_lhs
  // CHECK:    %[[TEN:.*]] = arith.constant 10
  // CHECK:    arith.shli %{{[0-9a-z]+}}, %[[TEN]]

  func.func @not_power_of_two(%x: i32) -> i32 {
    %10 = arith.constant 10 : i32
    %result = arith.muli %x, %10 : i32
    return %result : i32
  }
  // CHECK:      func.func @not_power_of_two
  // CHECK:        arith.constant
  // CHECK-NOT:    arith.shli
  // CHECK:        arith.muli

  func.func @not_constant(%x: i32, %y: i32) -> i32 {
    %result = arith.muli %x, %y : i32
    return %result : i32
  }
  // CHECK:      func.func @not_constant
  // CHECK-NOT:    arith.constant
  // CHECK-NOT:    arith.shli
  // CHECK:        arith.muli

  func.func @both_constants(%x: i32, %y: i32) -> i32 {
    %10 = arith.constant 10 : i32
    %32 = arith.constant 32 : i32
    %result = arith.muli %10, %32 : i32
    return %result : i32
  }
  // CHECK:      func.func @both_constants
  // CHECK:        arith.constant
  // CHECK-NOT:    arith.shli
}
