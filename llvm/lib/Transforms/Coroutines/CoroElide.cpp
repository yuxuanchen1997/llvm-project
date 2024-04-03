//===- CoroElide.cpp - Coroutine Frame Allocation Elision Pass ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/CoroElide.h"
#include "CoroInstr.h"
#include "CoroInternal.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "coro-elide"

STATISTIC(NumOfCoroElided, "The # of coroutine get elided.");

#ifndef NDEBUG
static cl::opt<std::string> CoroElideInfoOutputFilename(
    "coro-elide-info-output-file", cl::value_desc("filename"),
    cl::desc("File to record the coroutines got elided"), cl::Hidden);
#endif

namespace {
// Created on demand if the coro-elide pass has work to do.
struct Lowerer : coro::LowererBase {
  SmallVector<CoroIdInst *, 4> CoroIds;
  SmallVector<CoroBeginInst *, 1> CoroBegins;
  SmallVector<CoroAllocInst *, 1> CoroAllocs;
  SmallVector<Instruction *, 4> ResumeAddr;
  DenseMap<CoroBeginInst *, SmallVector<Instruction *, 4>> DestroyAddr;
  SmallPtrSet<const SwitchInst *, 4> CoroSuspendSwitches;

  Lowerer(Module &M) : LowererBase(M) {}

  void elideHeapAllocations(Function *F, uint64_t FrameSize, Align FrameAlign,
                            AAResults &AA);
  bool shouldElide(CoroIdInst *CoroId, DominatorTree &DT) const;
  void collectPostSplitCoroIds(Function *F);
  bool processCoroId(CoroIdInst *, AAResults &AA, DominatorTree &DT,
                     OptimizationRemarkEmitter &ORE);
  bool canValueEscape(Value *V, const SmallPtrSetImpl<BasicBlock *> &) const;
};
} // end anonymous namespace

// Go through the list of coro.subfn.addr intrinsics and replace them with the
// provided constant.
static void replaceWithConstant(Constant *Value,
                                SmallVectorImpl<Instruction *> &Users) {
  if (Users.empty())
    return;

  // See if we need to bitcast the constant to match the type of the intrinsic
  // being replaced. Note: All coro.subfn.addr intrinsics return the same type,
  // so we only need to examine the type of the first one in the list.
  Type *IntrTy = Users.front()->getType();
  Type *ValueTy = Value->getType();
  if (ValueTy != IntrTy) {
    // May need to tweak the function type to match the type expected at the
    // use site.
    assert(ValueTy->isPointerTy() && IntrTy->isPointerTy());
    Value = ConstantExpr::getBitCast(Value, IntrTy);
  }

  // Now the value type matches the type of the intrinsic. Replace them all!
  for (Instruction *I : Users)
    replaceAndRecursivelySimplify(I, Value);
}

// See if any operand of the call instruction references the coroutine frame.
static bool operandReferences(CallInst *CI, AllocaInst *Frame, AAResults &AA) {
  for (Value *Op : CI->operand_values())
    if (!AA.isNoAlias(Op, Frame))
      return true;
  return false;
}

// Look for any tail calls referencing the coroutine frame and remove tail
// attribute from them, since now coroutine frame resides on the stack and tail
// call implies that the function does not references anything on the stack.
// However if it's a musttail call, we cannot remove the tailcall attribute.
// It's safe to keep it there as the musttail call is for symmetric transfer,
// and by that point the frame should have been destroyed and hence not
// interfering with operands.
static void removeTailCallAttribute(AllocaInst *Frame, AAResults &AA) {
  Function &F = *Frame->getFunction();
  for (Instruction &I : instructions(F))
    if (auto *Call = dyn_cast<CallInst>(&I))
      if (Call->isTailCall() && operandReferences(Call, Frame, AA) &&
          !Call->isMustTailCall())
        Call->setTailCall(false);
}

// Given a resume function @f.resume(%f.frame* %frame), returns the size
// and expected alignment of %f.frame type.
static std::optional<std::pair<uint64_t, Align>>
getFrameLayout(Function *Resume) {
  // Pull information from the function attributes.
  auto Size = Resume->getParamDereferenceableBytes(0);
  if (!Size)
    return std::nullopt;
  return std::make_pair(Size, Resume->getParamAlign(0).valueOrOne());
}

// Finds first non alloca instruction in the entry block of a function.
static Instruction *getFirstNonAllocaInTheEntryBlock(Function *F) {
  for (Instruction &I : F->getEntryBlock())
    if (!isa<AllocaInst>(&I))
      return &I;
  llvm_unreachable("no terminator in the entry block");
}

#ifndef NDEBUG
static std::unique_ptr<raw_fd_ostream> getOrCreateLogFile() {
  assert(!CoroElideInfoOutputFilename.empty() &&
         "coro-elide-info-output-file shouldn't be empty");
  std::error_code EC;
  auto Result = std::make_unique<raw_fd_ostream>(CoroElideInfoOutputFilename,
                                                 EC, sys::fs::OF_Append);
  if (!EC)
    return Result;
  llvm::errs() << "Error opening coro-elide-info-output-file '"
               << CoroElideInfoOutputFilename << " for appending!\n";
  return std::make_unique<raw_fd_ostream>(2, false); // stderr.
}
#endif

// To elide heap allocations we need to suppress code blocks guarded by
// llvm.coro.alloc and llvm.coro.free instructions.
void Lowerer::elideHeapAllocations(Function *F, uint64_t FrameSize,
                                   Align FrameAlign, AAResults &AA) {
  LLVMContext &C = F->getContext();
  BasicBlock::iterator InsertPt =
      getFirstNonAllocaInTheEntryBlock(CoroIds.front()->getFunction())
          ->getIterator();

  // Replacing llvm.coro.alloc with false will suppress dynamic
  // allocation as it is expected for the frontend to generate the code that
  // looks like:
  //   id = coro.id(...)
  //   mem = coro.alloc(id) ? malloc(coro.size()) : 0;
  //   coro.begin(id, mem)
  auto *False = ConstantInt::getFalse(C);
  for (auto *CA : CoroAllocs) {
    CA->replaceAllUsesWith(False);
    CA->eraseFromParent();
  }

  // FIXME: Design how to transmit alignment information for every alloca that
  // is spilled into the coroutine frame and recreate the alignment information
  // here. Possibly we will need to do a mini SROA here and break the coroutine
  // frame into individual AllocaInst recreating the original alignment.
  const DataLayout &DL = F->getParent()->getDataLayout();
  auto FrameTy = ArrayType::get(Type::getInt8Ty(C), FrameSize);
  auto *Frame = new AllocaInst(FrameTy, DL.getAllocaAddrSpace(), "", InsertPt);
  Frame->setAlignment(FrameAlign);
  auto *FrameVoidPtr =
      new BitCastInst(Frame, PointerType::getUnqual(C), "vFrame", InsertPt);

  for (auto *CB : CoroBegins) {
    CB->replaceAllUsesWith(FrameVoidPtr);
    CB->eraseFromParent();
  }

  // Since now coroutine frame lives on the stack we need to make sure that
  // any tail call referencing it, must be made non-tail call.
  removeTailCallAttribute(Frame, AA);
}

bool Lowerer::canValueEscape(Value *V, const SmallPtrSetImpl<BasicBlock *> &TIs) const {
  SmallPtrSet<Value *, 10> ToVisit{V->user_begin(), V->user_end()};
  SmallPtrSet<Value *, 10> Visited;

  while (!ToVisit.empty()) {
    auto *VV = *ToVisit.begin();
    Visited.insert(VV);
    ToVisit.erase(VV);

    if (isa<CoroFreeInst, CoroSubFnInst, CoroSaveInst, SmugglePtrInst>(VV)) {
      continue;
    }

    if (auto *I = dyn_cast<Instruction>(VV)) {
      if (TIs.contains(I->getParent())) {
        return true;
      }
    }

    for (auto *U : VV->users()) {
      if (!Visited.contains(U)) {
        ToVisit.insert(U);
      }
    }
  }

  return false;
}

bool Lowerer::shouldElide(CoroIdInst *CoroId, DominatorTree &DT) const {
  // If no CoroAllocs, we cannot suppress allocation, so elision is not
  // possible.
  if (CoroAllocs.empty())
    return false;

  auto *ContainingFunction = CoroId->getFunction();

  // Check that for every coro.begin there is at least one coro.destroy directly
  // referencing the SSA value of that coro.begin along each
  // non-exceptional path.
  // If the value escaped, then coro.destroy would have been referencing a
  // memory location storing that value and not the virtual register.

  SmallPtrSet<BasicBlock *, 8> Terminators;
  // First gather all of the terminators for the function.
  // Consider the final coro.suspend as the real terminator when the current
  // function is a coroutine.
  for (BasicBlock &B : *ContainingFunction) {
    auto *TI = B.getTerminator();

    if (TI->getNumSuccessors() != 0 || isa<UnreachableInst>(TI))
      continue;

    Terminators.insert(&B);
  }

  SmallPtrSet<CoroBeginInst *, 8> CorrespondingCoroBegins;
  for (User *U: CoroId->users()) {
    if (auto *CBI = dyn_cast<CoroBeginInst>(U)) {
      CorrespondingCoroBegins.insert(CBI);
    }
  }

  // Filter out the coro.destroy that lie along exceptional paths.
  SmallPtrSet<CoroBeginInst *, 8> ReferencedCoroBegins;

  for (const auto CB : CorrespondingCoroBegins) {
    if (!canValueEscape(CB, Terminators))
      ReferencedCoroBegins.insert(CB);
  }

  // If size of the set is the same as total number of coro.begin, that means we
  // found a coro.free or coro.destroy referencing each coro.begin, so we can
  // perform heap elision.
  return ReferencedCoroBegins.size() == CorrespondingCoroBegins.size();
}

void Lowerer::collectPostSplitCoroIds(Function *F) {
  CoroIds.clear();
  CoroSuspendSwitches.clear();
  for (auto &I : instructions(F)) {
    if (auto *CII = dyn_cast<CoroIdInst>(&I))
      if (CII->getInfo().isPostSplit())
        // If it is the coroutine itself, don't touch it.
        if (CII->getCoroutine() != CII->getFunction())
          CoroIds.push_back(CII);

    // Consider case like:
    // %0 = call i8 @llvm.coro.suspend(...)
    // switch i8 %0, label %suspend [i8 0, label %resume
    //                              i8 1, label %cleanup]
    // and collect the SwitchInsts which are used by escape analysis later.
    if (auto *CSI = dyn_cast<CoroSuspendInst>(&I))
      if (CSI->hasOneUse() && isa<SwitchInst>(CSI->use_begin()->getUser())) {
        SwitchInst *SWI = cast<SwitchInst>(CSI->use_begin()->getUser());
        if (SWI->getNumCases() == 2)
          CoroSuspendSwitches.insert(SWI);
      }
  }
}

bool Lowerer::processCoroId(CoroIdInst *CoroId, AAResults &AA,
                            DominatorTree &DT, OptimizationRemarkEmitter &ORE) {
  CoroBegins.clear();
  CoroAllocs.clear();
  ResumeAddr.clear();
  DestroyAddr.clear();

  // Collect all coro.begin and coro.allocs associated with this coro.id.
  for (User *U : CoroId->users()) {
    if (auto *CB = dyn_cast<CoroBeginInst>(U))
      CoroBegins.push_back(CB);
    else if (auto *CA = dyn_cast<CoroAllocInst>(U))
      CoroAllocs.push_back(CA);
  }

  // Collect all coro.subfn.addrs associated with coro.begin.
  // Note, we only devirtualize the calls if their coro.subfn.addr refers to
  // coro.begin directly. If we run into cases where this check is too
  // conservative, we can consider relaxing the check.
  for (CoroBeginInst *CB : CoroBegins) {
    for (User *U : CB->users()) {
      if (auto *II = dyn_cast<CoroSubFnInst>(U)) {
        switch (II->getIndex()) {
        case CoroSubFnInst::ResumeIndex:
          ResumeAddr.push_back(II);
          break;
        case CoroSubFnInst::DestroyIndex:
          DestroyAddr[CB].push_back(II);
          break;
        default:
          llvm_unreachable("unexpected coro.subfn.addr constant");
        }
      }

      if (auto *SPI = dyn_cast<SmugglePtrInst>(U)) {
        DestroyAddr[CB].push_back(SPI);
      }
    }
  }

  // PostSplit coro.id refers to an array of subfunctions in its Info
  // argument.
  ConstantArray *Resumers = CoroId->getInfo().Resumers;
  assert(Resumers && "PostSplit coro.id Info argument must refer to an array"
                     "of coroutine subfunctions");
  auto *ResumeAddrConstant =
      Resumers->getAggregateElement(CoroSubFnInst::ResumeIndex);

  replaceWithConstant(ResumeAddrConstant, ResumeAddr);

  bool ShouldElide = shouldElide(CoroId, DT);
  llvm::dbgs() << "ShouldElide: " << ShouldElide << " "; CoroId->dump();
  if (!ShouldElide)
    ORE.emit([&]() {
      if (auto FrameSizeAndAlign =
              getFrameLayout(cast<Function>(ResumeAddrConstant)))
        return OptimizationRemarkMissed(DEBUG_TYPE, "CoroElide", CoroId)
               << "'" << ore::NV("callee", CoroId->getCoroutine()->getName())
               << "' not elided in '"
               << ore::NV("caller", CoroId->getFunction()->getName())
               << "' (frame_size="
               << ore::NV("frame_size", FrameSizeAndAlign->first) << ", align="
               << ore::NV("align", FrameSizeAndAlign->second.value()) << ")";
      else
        return OptimizationRemarkMissed(DEBUG_TYPE, "CoroElide", CoroId)
               << "'" << ore::NV("callee", CoroId->getCoroutine()->getName())
               << "' not elided in '"
               << ore::NV("caller", CoroId->getFunction()->getName())
               << "' (frame_size=unknown, align=unknown)";
    });


  if (ShouldElide) {
    auto *DestroyAddrConstant = Resumers->getAggregateElement(
        ShouldElide ? CoroSubFnInst::CleanupIndex : CoroSubFnInst::DestroyIndex);

    for (auto &It : DestroyAddr)
      replaceWithConstant(DestroyAddrConstant, It.second);

    if (auto FrameSizeAndAlign =
            getFrameLayout(cast<Function>(ResumeAddrConstant))) {
      elideHeapAllocations(CoroId->getFunction(), FrameSizeAndAlign->first,
                           FrameSizeAndAlign->second, AA);
      coro::replaceCoroFree(CoroId, /*Elide=*/true);
      NumOfCoroElided++;
#ifndef NDEBUG
      if (!CoroElideInfoOutputFilename.empty())
        *getOrCreateLogFile()
            << "Elide " << CoroId->getCoroutine()->getName() << " in "
            << CoroId->getFunction()->getName() << "\n";
#endif
      ORE.emit([&]() {
        return OptimizationRemark(DEBUG_TYPE, "CoroElide", CoroId)
               << "'" << ore::NV("callee", CoroId->getCoroutine()->getName())
               << "' elided in '"
               << ore::NV("caller", CoroId->getFunction()->getName())
               << "' (frame_size="
               << ore::NV("frame_size", FrameSizeAndAlign->first) << ", align="
               << ore::NV("align", FrameSizeAndAlign->second.value()) << ")";
      });
    } else {
      ORE.emit([&]() {
        return OptimizationRemarkMissed(DEBUG_TYPE, "CoroElide", CoroId)
               << "'" << ore::NV("callee", CoroId->getCoroutine()->getName())
               << "' not elided in '"
               << ore::NV("caller", CoroId->getFunction()->getName())
               << "' (frame_size=unknown, align=unknown)";
      });
    }
  }

  return true;
}

static bool declaresCoroElideIntrinsics(Module &M) {
  return coro::declaresIntrinsics(M, {"llvm.coro.id", "llvm.coro.id.async"});
}

PreservedAnalyses CoroElidePass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &M = *F.getParent();
  if (!declaresCoroElideIntrinsics(M))
    return PreservedAnalyses::all();

  Lowerer L(M);
  L.CoroIds.clear();
  L.collectPostSplitCoroIds(&F);
  // If we did not find any coro.id, there is nothing to do.
  if (L.CoroIds.empty())
    return PreservedAnalyses::all();

  AAResults &AA = AM.getResult<AAManager>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  bool Changed = false;
  for (auto *CII : L.CoroIds)
    Changed |= L.processCoroId(CII, AA, DT, ORE);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
