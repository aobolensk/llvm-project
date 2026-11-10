; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs -stop-after=spirv-postlegalizer %s -o %t.mir
; RUN: FileCheck %s --input-file=%t.mir
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -run-pass=none %t.mir -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs %s -o %t.ref.s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs -start-after=spirv-postlegalizer %t.mir -o %t.s
; RUN: diff %t.ref.s %t.s

; CHECK-LABEL: name: foo
; CHECK: machineFunctionInfo:
; CHECK-NEXT: vregTypes:
; CHECK-NEXT: - { reg: '[[P:%[0-9]+]]', type: '[[PTR:%[0-9]+]]' }
; CHECK-NEXT: - { reg: '[[A:%[0-9]+]]', type: '[[I32:%[0-9]+]]' }
; CHECK-NEXT: - { reg: '[[B:%[0-9]+]]', type: '[[I32]]' }
; CHECK-NEXT: - { reg: '[[SUM:%[0-9]+]]', type: '[[I32]]' }
; CHECK-NEXT: - { reg: '[[ADD:%[0-9]+]]', type: '[[I32]]' }
; CHECK-NEXT: lastInsertedType: '[[FN:%[0-9]+]]'
; CHECK-NEXT: body:
; CHECK-DAG: [[PTR]]:type(s64) = OpTypePointer 5, [[I32]](s64)
; CHECK-DAG: [[FN]]:type(s64) = OpTypeFunction [[I32]](s64), [[PTR]](s64), [[I32]](s64), [[I32]](s64)
; CHECK-DAG: [[I32]]:type(s64) = OpTypeInt 32, 0
; CHECK-DAG: [[P]]:pid(p1) = OpFunctionParameter [[PTR]](s64)
; CHECK-DAG: [[A]]:iid(s32) = OpFunctionParameter [[I32]](s64)
; CHECK-DAG: [[B]]:iid(s32) = OpFunctionParameter [[I32]](s64)
; CHECK-DAG: [[ADD]]:iid(s32) = G_ADD [[A]], [[B]]
; CHECK-DAG: [[SUM]]:iid(s32) = ASSIGN_TYPE [[ADD]](s32), [[I32]](s64)

define spir_func i32 @foo(ptr addrspace(1) %p, i32 %a, i32 %b) {
  %s = add i32 %a, %b
  store i32 %s, ptr addrspace(1) %p
  ret i32 %s
}
