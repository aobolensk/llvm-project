; RUN: llc -O0 -mtriple=spirv32-unknown-unknown -verify-machineinstrs %s -filetype=obj -o - | od -A n -t x4 -v -w4096 | FileCheck %s

;; fp128 is encoded as four 32-bit literal words. Check that
;; SPIRVGlobalRegistry::createConstFP emits the full 128-bit value (not
;; truncated to its low 64 bits) in the binary OpConstant instruction.
;; The textual/OpConstantF printer path is not exercised here: fp128 is
;; not valid standard SPIR-V, so spirv-dis/spirv-val cannot process it.

define fp128 @getConstantFP128() {
  ret fp128 0xL00000000000000004001000000000000 ; 4.0
}

;; OpConstant %float128 %4, wordcount 7, literal words 0, 0, 0, 0x40010000.
; CHECK: 0007002b {{[0-9a-f]+}} {{[0-9a-f]+}} 00000000 00000000 00000000 40010000
