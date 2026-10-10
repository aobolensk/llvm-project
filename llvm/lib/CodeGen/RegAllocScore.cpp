//===- RegAllocScore.cpp - evaluate regalloc policy quality ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Calculate a measure of the register allocation policy quality. This is used
/// to construct a reward for the training of the ML-driven allocation policy.
/// Currently, the score is the sum of the machine basic block frequency-weighed
/// number of loads, stores, copies, and remat instructions, each factored with
/// a relative weight.
//===----------------------------------------------------------------------===//

#include "RegAllocScore.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<double> CopyWeight("regalloc-copy-weight", cl::init(0.2),
                                  cl::Hidden);
static cl::opt<double> LoadWeight("regalloc-load-weight", cl::init(4.0),
                                  cl::Hidden);
static cl::opt<double> StoreWeight("regalloc-store-weight", cl::init(1.0),
                                   cl::Hidden);
static cl::opt<double> CheapRematWeight("regalloc-cheap-remat-weight",
                                        cl::init(0.2), cl::Hidden);
static cl::opt<double> ExpensiveRematWeight("regalloc-expensive-remat-weight",
                                            cl::init(1.0), cl::Hidden);
static cl::opt<unsigned> WidthUnit(
    "regalloc-score-width-unit", cl::init(0), cl::Hidden,
    cl::desc("If nonzero, weight copies, loads and stores by their width in "
             "units of this many bits"));

#define DEBUG_TYPE "regalloc-score"

RegAllocScore &RegAllocScore::operator+=(const RegAllocScore &Other) {
  CopyCounts += Other.copyCounts();
  LoadCounts += Other.loadCounts();
  StoreCounts += Other.storeCounts();
  LoadStoreCounts += Other.loadStoreCounts();
  CheapRematCounts += Other.cheapRematCounts();
  ExpensiveRematCounts += Other.expensiveRematCounts();
  return *this;
}

bool RegAllocScore::operator==(const RegAllocScore &Other) const {
  return copyCounts() == Other.copyCounts() &&
         loadCounts() == Other.loadCounts() &&
         storeCounts() == Other.storeCounts() &&
         loadStoreCounts() == Other.loadStoreCounts() &&
         cheapRematCounts() == Other.cheapRematCounts() &&
         expensiveRematCounts() == Other.expensiveRematCounts();
}

bool RegAllocScore::operator!=(const RegAllocScore &Other) const {
  return !(*this == Other);
}

double RegAllocScore::getScore() const {
  double Ret = 0.0;
  Ret += CopyWeight * copyCounts();
  Ret += LoadWeight * loadCounts();
  Ret += StoreWeight * storeCounts();
  Ret += (LoadWeight + StoreWeight) * loadStoreCounts();
  Ret += CheapRematWeight * cheapRematCounts();
  Ret += ExpensiveRematWeight * expensiveRematCounts();

  return Ret;
}

RegAllocScore
llvm::calculateRegAllocScore(const MachineFunction &MF,
                             const MachineBlockFrequencyInfo &MBFI) {
  return calculateRegAllocScore(
      MF,
      [&](const MachineBasicBlock &MBB) {
        return MBFI.getBlockFreqRelativeToEntryBlock(&MBB);
      },
      [&](const MachineInstr &MI) {
        return MF.getSubtarget().getInstrInfo()->isReMaterializable(MI);
      },
      WidthUnit);
}

/// Number of UnitBits units covered by Bits, at least 1. Unknown widths are
/// passed as 0.
static double getWidthFactor(uint64_t Bits, unsigned UnitBits) {
  return std::max<uint64_t>(1, divideCeil(Bits, UnitBits));
}

static double getCopyWidthFactor(const MachineInstr &MI, unsigned UnitBits) {
  if (!UnitBits)
    return 1.0;
  const MachineFunction &MF = *MI.getMF();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const MachineOperand &Dst = MI.getOperand(0);
  if (unsigned SubIdx = Dst.getSubReg()) {
    // -1 means the size depends on the register.
    unsigned Bits = TRI.getSubRegIdxSize(SubIdx);
    return getWidthFactor(Bits == ~0u ? 0 : Bits, UnitBits);
  }
  Register Reg = Dst.getReg();
  TypeSize Size = TypeSize::getFixed(0);
  if (Reg.isPhysical()) {
    if (const TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg))
      Size = TRI.getRegSizeInBits(*RC);
  } else {
    Size = TRI.getRegSizeInBits(Reg, MF.getRegInfo());
  }
  return getWidthFactor(Size.isScalable() ? 0 : Size.getFixedValue(), UnitBits);
}

static double getMemWidthFactor(const MachineInstr &MI, unsigned UnitBits) {
  if (!UnitBits)
    return 1.0;
  uint64_t Bits = 0;
  for (const MachineMemOperand *MMO : MI.memoperands()) {
    LocationSize Size = MMO->getSizeInBits();
    if (Size.isPrecise() && !Size.isScalable())
      Bits = std::max(Bits, Size.getValue().getFixedValue());
  }
  return getWidthFactor(Bits, UnitBits);
}

RegAllocScore llvm::calculateRegAllocScore(
    const MachineFunction &MF,
    llvm::function_ref<double(const MachineBasicBlock &)> GetBBFreq,
    llvm::function_ref<bool(const MachineInstr &)> IsTriviallyRematerializable,
    unsigned WidthUnitBits) {
  RegAllocScore Total;

  for (const MachineBasicBlock &MBB : MF) {
    double BlockFreqRelativeToEntrypoint = GetBBFreq(MBB);
    RegAllocScore MBBScore;

    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugInstr() || MI.isKill() || MI.isInlineAsm()) {
        continue;
      }
      if (MI.isCopy()) {
        MBBScore.onCopy(BlockFreqRelativeToEntrypoint *
                        getCopyWidthFactor(MI, WidthUnitBits));
      } else if (IsTriviallyRematerializable(MI)) {
        if (MI.getDesc().isAsCheapAsAMove()) {
          MBBScore.onCheapRemat(BlockFreqRelativeToEntrypoint);
        } else {
          MBBScore.onExpensiveRemat(BlockFreqRelativeToEntrypoint);
        }
      } else if (MI.mayLoad() || MI.mayStore()) {
        double Freq = BlockFreqRelativeToEntrypoint *
                      getMemWidthFactor(MI, WidthUnitBits);
        if (!MI.mayStore())
          MBBScore.onLoad(Freq);
        else if (!MI.mayLoad())
          MBBScore.onStore(Freq);
        else
          MBBScore.onLoadStore(Freq);
      }
    }
    Total += MBBScore;
  }
  return Total;
}
