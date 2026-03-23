// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

#define __device__ __attribute__((device))

// CHECK-LABEL: @_Z4testPi(
// CHECK: call addrspace(4) i32 @llvm.spv.workgroup.size.i32(i32 0)
// CHECK: call addrspace(4) i32 @llvm.spv.workgroup.size.i32(i32 1)
// CHECK: call addrspace(4) i32 @llvm.spv.workgroup.size.i32(i32 2)
// CHECK-NOT: @llvm.amdgcn.implicitarg.ptr
// CHECK-NOT: @llvm.amdgcn.dispatch.ptr
__device__ void test(int *out) {
  out[0] = __builtin_amdgcn_workgroup_size_x();
  out[1] = __builtin_amdgcn_workgroup_size_y();
  out[2] = __builtin_amdgcn_workgroup_size_z();
}
