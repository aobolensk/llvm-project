; A call to a declared-but-undefined function with an odd-width integer argument.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#I32:]] = OpTypeInt 32 0
; CHECK: OpFunctionCall %[[#I32]]

declare spir_func i32 @undefined_callee(i36)

define spir_kernel void @fuzz_kernel(ptr addrspace(1) %out, i36 %n) {
entry:
  %r = call spir_func i32 @undefined_callee(i36 0)
  store i32 %r, ptr addrspace(1) %out
  ret void
}
