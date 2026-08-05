//===- ComplexToSPIRV.cpp - Complex to SPIR-V Patterns --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Complex dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToSPIRV/ComplexToSPIRV.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "complex-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

/// Creates a scalar floating-point constant of the given value.
Value createFPConstant(OpBuilder &builder, Location loc, Type type,
                       double value) {
  return spirv::ConstantOp::create(
      builder, loc, type, builder.getFloatAttr(cast<FloatType>(type), value));
}

/// Computes the SPIR-V equivalent of complex square root or reciprocal square
/// root from its real and imaginary components, using the trigonometric form
/// sqrt(re + i*im) = sqrt(abs) * (cos(angle/2) + i*sin(angle/2))
/// rsqrt(re + i*im) = 1/sqrt(abs) * (cos(-angle/2) + i*sin(-angle/2))
/// where abs = sqrt(re^2 + im^2), angle = atan2(im, re), and FinalOp computes
/// either sqrt or 1/sqrt of abs depending on angleScale.
template <typename SqrtOp, typename SinOp, typename CosOp, typename Atan2Op,
          typename FinalOp>
std::pair<Value, Value>
createComplexSqrtOrRsqrt(ConversionPatternRewriter &rewriter, Location loc,
                         Value re, Value im, double angleScale) {
  Value reSq = spirv::FMulOp::create(rewriter, loc, re, re);
  Value imSq = spirv::FMulOp::create(rewriter, loc, im, im);
  Value sum = spirv::FAddOp::create(rewriter, loc, reSq, imSq);
  Value absVal = SqrtOp::create(rewriter, loc, sum);
  Value finalAbs = FinalOp::create(rewriter, loc, absVal);
  Value angle = Atan2Op::create(rewriter, loc, im, re);
  Value scale = createFPConstant(rewriter, loc, re.getType(), angleScale);
  Value scaledAngle = spirv::FMulOp::create(rewriter, loc, angle, scale);
  Value cosVal = CosOp::create(rewriter, loc, scaledAngle);
  Value sinVal = SinOp::create(rewriter, loc, scaledAngle);
  Value resultRe = spirv::FMulOp::create(rewriter, loc, finalAbs, cosVal);
  Value resultIm = spirv::FMulOp::create(rewriter, loc, finalAbs, sinVal);
  return {resultRe, resultIm};
}

struct ConstantOpPattern final : OpConversionPattern<complex::ConstantOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto spirvType =
        getTypeConverter()->convertType<ShapedType>(constOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(constOp,
                                         "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
        constOp, spirvType,
        DenseElementsAttr::get(spirvType, constOp.getValue().getValue()));
    return success();
  }
};

struct CreateOpPattern final : OpConversionPattern<complex::CreateOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::CreateOp createOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(createOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(createOp,
                                         "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        createOp, spirvType, adaptor.getOperands());
    return success();
  }
};

struct ReOpPattern final : OpConversionPattern<complex::ReOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::ReOp reOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(reOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(reOp, "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        reOp, adaptor.getComplex(), llvm::ArrayRef(0));
    return success();
  }
};

struct ImOpPattern final : OpConversionPattern<complex::ImOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::ImOp imOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(imOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(imOp, "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        imOp, adaptor.getComplex(), llvm::ArrayRef(1));
    return success();
  }
};

template <typename ComplexOp, typename SPIRVOp>
struct ElementwiseBinaryOpPattern final : OpConversionPattern<ComplexOp> {
  using OpConversionPattern<ComplexOp>::OpConversionPattern;
  using OpAdaptor = typename ComplexOp::Adaptor;

  LogicalResult
  matchAndRewrite(ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value lhsRe = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {0});
    Value lhsIm = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {1});
    Value rhsRe = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {0});
    Value rhsIm = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {1});

    Value resultRe = SPIRVOp::create(rewriter, loc, lhsRe, rhsRe);
    Value resultIm = SPIRVOp::create(rewriter, loc, lhsIm, rhsIm);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

template <typename ComplexOp, typename SPIRVCompareOp, typename SPIRVCombinerOp>
struct ComparisonOpPattern final : OpConversionPattern<ComplexOp> {
  using OpConversionPattern<ComplexOp>::OpConversionPattern;
  using OpAdaptor = typename ComplexOp::Adaptor;

  LogicalResult
  matchAndRewrite(ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value lhsRe = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {0});
    Value lhsIm = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {1});
    Value rhsRe = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {0});
    Value rhsIm = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {1});

    Value cmpRe = SPIRVCompareOp::create(rewriter, loc, lhsRe, rhsRe);
    Value cmpIm = SPIRVCompareOp::create(rewriter, loc, lhsIm, rhsIm);

    rewriter.replaceOpWithNewOp<SPIRVCombinerOp>(op, spirvType, cmpRe, cmpIm);
    return success();
  }
};

struct MulOpPattern final : OpConversionPattern<complex::MulOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value a = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {0});
    Value b = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {1});
    Value c = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {0});
    Value d = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {1});

    Value ac = spirv::FMulOp::create(rewriter, loc, a, c);
    Value bd = spirv::FMulOp::create(rewriter, loc, b, d);
    Value ad = spirv::FMulOp::create(rewriter, loc, a, d);
    Value bc = spirv::FMulOp::create(rewriter, loc, b, c);
    Value resultRe = spirv::FSubOp::create(rewriter, loc, ac, bd);
    Value resultIm = spirv::FAddOp::create(rewriter, loc, ad, bc);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

template <typename SqrtOp>
struct AbsOpPattern final : OpConversionPattern<complex::AbsOp> {
  using OpConversionPattern<complex::AbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value reSq = spirv::FMulOp::create(rewriter, loc, re, re);
    Value imSq = spirv::FMulOp::create(rewriter, loc, im, im);
    Value sum = spirv::FAddOp::create(rewriter, loc, reSq, imSq);

    rewriter.replaceOpWithNewOp<SqrtOp>(op, sum);
    return success();
  }
};

template <typename ComplexOp, bool NegateReal>
struct NegationOpPattern final : OpConversionPattern<ComplexOp> {
  using OpConversionPattern<ComplexOp>::OpConversionPattern;
  using OpAdaptor = typename ComplexOp::Adaptor;

  LogicalResult
  matchAndRewrite(ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value resultRe =
        NegateReal ? spirv::FNegateOp::create(rewriter, loc, re) : re;
    Value resultIm = spirv::FNegateOp::create(rewriter, loc, im);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

struct DivOpPattern final : OpConversionPattern<complex::DivOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(complex::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value a = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {0});
    Value b = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {1});
    Value c = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {0});
    Value d = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {1});

    Value ac = spirv::FMulOp::create(rewriter, loc, a, c);
    Value bd = spirv::FMulOp::create(rewriter, loc, b, d);
    Value bc = spirv::FMulOp::create(rewriter, loc, b, c);
    Value ad = spirv::FMulOp::create(rewriter, loc, a, d);
    Value cc = spirv::FMulOp::create(rewriter, loc, c, c);
    Value dd = spirv::FMulOp::create(rewriter, loc, d, d);
    Value denom = spirv::FAddOp::create(rewriter, loc, cc, dd);
    Value numRe = spirv::FAddOp::create(rewriter, loc, ac, bd);
    Value numIm = spirv::FSubOp::create(rewriter, loc, bc, ad);
    Value resultRe = spirv::FDivOp::create(rewriter, loc, numRe, denom);
    Value resultIm = spirv::FDivOp::create(rewriter, loc, numIm, denom);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

template <typename Atan2Op>
struct AngleOpPattern final : OpConversionPattern<complex::AngleOp> {
  using OpConversionPattern<complex::AngleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::AngleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    rewriter.replaceOpWithNewOp<Atan2Op>(op, im, re);
    return success();
  }
};

/// Computes complex.sign as z / abs(z), selecting a zero result when z is
/// zero to avoid a division by zero.
template <typename SqrtOp>
struct SignOpPattern final : OpConversionPattern<complex::SignOp> {
  using OpConversionPattern<complex::SignOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::SignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value reSq = spirv::FMulOp::create(rewriter, loc, re, re);
    Value imSq = spirv::FMulOp::create(rewriter, loc, im, im);
    Value sum = spirv::FAddOp::create(rewriter, loc, reSq, imSq);
    Value abs = SqrtOp::create(rewriter, loc, sum);

    Value signRe = spirv::FDivOp::create(rewriter, loc, re, abs);
    Value signIm = spirv::FDivOp::create(rewriter, loc, im, abs);

    Value zero = createFPConstant(rewriter, loc, re.getType(), 0.0);
    Value reIsZero = spirv::FOrdEqualOp::create(rewriter, loc, re, zero);
    Value imIsZero = spirv::FOrdEqualOp::create(rewriter, loc, im, zero);
    Value isZero =
        spirv::LogicalAndOp::create(rewriter, loc, reIsZero, imIsZero);

    // Select per component: spirv.Select with a scalar condition and a
    // composite result requires SPIR-V 1.4, so operate on scalars here and
    // assemble the result afterwards.
    Value resultRe = spirv::SelectOp::create(rewriter, loc, isZero, re, signRe);
    Value resultIm = spirv::SelectOp::create(rewriter, loc, isZero, im, signIm);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

/// Common building blocks for complex.sin/cos, following the identities
/// sin(x + iy) = sin(x) * (0.5*exp(y) + 0.5/exp(y))
///              + i * cos(x) * (0.5*exp(y) - 0.5/exp(y))
/// cos(x + iy) = cos(x) * (0.5/exp(y) + 0.5*exp(y))
///              + i * sin(x) * (0.5/exp(y) - 0.5*exp(y))
template <typename ComplexOp, typename SinOp, typename CosOp, typename ExpOp>
struct TrigonometricOpPattern : OpConversionPattern<ComplexOp> {
  using OpConversionPattern<ComplexOp>::OpConversionPattern;
  using OpAdaptor = typename ComplexOp::Adaptor;

  LogicalResult
  matchAndRewrite(ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value half = createFPConstant(rewriter, loc, re.getType(), 0.5);
    Value exp = ExpOp::create(rewriter, loc, im);
    Value scaledExp = spirv::FMulOp::create(rewriter, loc, half, exp);
    Value reciprocalExp = spirv::FDivOp::create(rewriter, loc, half, exp);
    Value sin = SinOp::create(rewriter, loc, re);
    Value cos = CosOp::create(rewriter, loc, re);

    auto [resultRe, resultIm] =
        combine(rewriter, loc, scaledExp, reciprocalExp, sin, cos);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }

  virtual std::pair<Value, Value> combine(ConversionPatternRewriter &rewriter,
                                          Location loc, Value scaledExp,
                                          Value reciprocalExp, Value sin,
                                          Value cos) const = 0;
};

template <typename SinOp, typename CosOp, typename ExpOp>
struct SinOpPattern final
    : TrigonometricOpPattern<complex::SinOp, SinOp, CosOp, ExpOp> {
  using TrigonometricOpPattern<complex::SinOp, SinOp, CosOp,
                               ExpOp>::TrigonometricOpPattern;

  std::pair<Value, Value> combine(ConversionPatternRewriter &rewriter,
                                  Location loc, Value scaledExp,
                                  Value reciprocalExp, Value sin,
                                  Value cos) const override {
    Value sum = spirv::FAddOp::create(rewriter, loc, scaledExp, reciprocalExp);
    Value resultRe = spirv::FMulOp::create(rewriter, loc, sum, sin);
    Value diff = spirv::FSubOp::create(rewriter, loc, scaledExp, reciprocalExp);
    Value resultIm = spirv::FMulOp::create(rewriter, loc, diff, cos);
    return {resultRe, resultIm};
  }
};

template <typename SinOp, typename CosOp, typename ExpOp>
struct CosOpPattern final
    : TrigonometricOpPattern<complex::CosOp, SinOp, CosOp, ExpOp> {
  using TrigonometricOpPattern<complex::CosOp, SinOp, CosOp,
                               ExpOp>::TrigonometricOpPattern;

  std::pair<Value, Value> combine(ConversionPatternRewriter &rewriter,
                                  Location loc, Value scaledExp,
                                  Value reciprocalExp, Value sin,
                                  Value cos) const override {
    Value sum = spirv::FAddOp::create(rewriter, loc, reciprocalExp, scaledExp);
    Value resultRe = spirv::FMulOp::create(rewriter, loc, sum, cos);
    Value diff = spirv::FSubOp::create(rewriter, loc, reciprocalExp, scaledExp);
    Value resultIm = spirv::FMulOp::create(rewriter, loc, diff, sin);
    return {resultRe, resultIm};
  }
};

/// Computes complex.exp as exp(x) * (cos(y) + i*sin(y)).
template <typename ExpOp, typename SinOp, typename CosOp>
struct ExpOpPattern final : OpConversionPattern<complex::ExpOp> {
  using OpConversionPattern<complex::ExpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value expRe = ExpOp::create(rewriter, loc, re);
    Value sinIm = SinOp::create(rewriter, loc, im);
    Value cosIm = CosOp::create(rewriter, loc, im);
    Value resultRe = spirv::FMulOp::create(rewriter, loc, expRe, cosIm);
    Value resultIm = spirv::FMulOp::create(rewriter, loc, expRe, sinIm);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

/// Computes complex.expm1(z) as exp(re) * cos(im) - 1 + i * exp(re) * sin(im).
template <typename ExpOp, typename SinOp, typename CosOp>
struct Expm1OpPattern final : OpConversionPattern<complex::Expm1Op> {
  using OpConversionPattern<complex::Expm1Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::Expm1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value expRe = ExpOp::create(rewriter, loc, re);
    Value sinIm = SinOp::create(rewriter, loc, im);
    Value cosIm = CosOp::create(rewriter, loc, im);
    Value one = createFPConstant(rewriter, loc, re.getType(), 1.0);
    Value expReCosIm = spirv::FMulOp::create(rewriter, loc, expRe, cosIm);
    Value resultRe = spirv::FSubOp::create(rewriter, loc, expReCosIm, one);
    Value resultIm = spirv::FMulOp::create(rewriter, loc, expRe, sinIm);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

/// Computes complex.log(z) as log(abs(z)) + i*atan2(im, re).
template <typename SqrtOp, typename LogOp, typename Atan2Op>
struct LogOpPattern final : OpConversionPattern<complex::LogOp> {
  using OpConversionPattern<complex::LogOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::LogOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value reSq = spirv::FMulOp::create(rewriter, loc, re, re);
    Value imSq = spirv::FMulOp::create(rewriter, loc, im, im);
    Value sum = spirv::FAddOp::create(rewriter, loc, reSq, imSq);
    Value abs = SqrtOp::create(rewriter, loc, sum);

    Value resultRe = LogOp::create(rewriter, loc, abs);
    Value resultIm = Atan2Op::create(rewriter, loc, im, re);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

/// Computes complex.log1p(z) as log(abs(1 + z)) + i*atan2(im, 1 + re).
template <typename SqrtOp, typename LogOp, typename Atan2Op>
struct Log1pOpPattern final : OpConversionPattern<complex::Log1pOp> {
  using OpConversionPattern<complex::Log1pOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::Log1pOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value one = createFPConstant(rewriter, loc, re.getType(), 1.0);
    Value onePlusRe = spirv::FAddOp::create(rewriter, loc, one, re);

    Value reSq = spirv::FMulOp::create(rewriter, loc, onePlusRe, onePlusRe);
    Value imSq = spirv::FMulOp::create(rewriter, loc, im, im);
    Value sum = spirv::FAddOp::create(rewriter, loc, reSq, imSq);
    Value abs = SqrtOp::create(rewriter, loc, sum);

    Value resultRe = LogOp::create(rewriter, loc, abs);
    Value resultIm = Atan2Op::create(rewriter, loc, im, onePlusRe);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

/// Computes complex.sqrt(z) via the trigonometric form of square root.
template <typename SqrtOp, typename SinOp, typename CosOp, typename Atan2Op>
struct SqrtOpPattern final : OpConversionPattern<complex::SqrtOp> {
  using OpConversionPattern<complex::SqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    auto [resultRe, resultIm] =
        createComplexSqrtOrRsqrt<SqrtOp, SinOp, CosOp, Atan2Op, SqrtOp>(
            rewriter, loc, re, im, /*angleScale=*/0.5);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

/// Computes complex.rsqrt(z) via the trigonometric form of reciprocal square
/// root.
template <typename SqrtOp, typename SinOp, typename CosOp, typename Atan2Op,
          typename RsqrtOp>
struct RsqrtOpPattern final : OpConversionPattern<complex::RsqrtOp> {
  using OpConversionPattern<complex::RsqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    auto [resultRe, resultIm] =
        createComplexSqrtOrRsqrt<SqrtOp, SinOp, CosOp, Atan2Op, RsqrtOp>(
            rewriter, loc, re, im, /*angleScale=*/-0.5);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

/// Computes complex.tan/tanh via the real/imaginary component identities
/// tan(x + iy) = (sin(2x) + i*sinh(2y)) / (cos(2x) + cosh(2y))
/// tanh(x + iy) = (sinh(2x) + i*sin(2y)) / (cosh(2x) + cos(2y))
template <typename ComplexOp, typename RealSinOp, typename RealCosOp,
          typename ImagSinOp, typename ImagCosOp>
struct TanTanhOpPattern final : OpConversionPattern<ComplexOp> {
  using OpConversionPattern<ComplexOp>::OpConversionPattern;
  using OpAdaptor = typename ComplexOp::Adaptor;

  LogicalResult
  matchAndRewrite(ComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value complexVal = adaptor.getComplex();

    Value re =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {0});
    Value im =
        spirv::CompositeExtractOp::create(rewriter, loc, complexVal, {1});

    Value two = createFPConstant(rewriter, loc, re.getType(), 2.0);
    Value twoRe = spirv::FMulOp::create(rewriter, loc, re, two);
    Value twoIm = spirv::FMulOp::create(rewriter, loc, im, two);

    Value realSin = RealSinOp::create(rewriter, loc, twoRe);
    Value realCos = RealCosOp::create(rewriter, loc, twoRe);
    Value imagSin = ImagSinOp::create(rewriter, loc, twoIm);
    Value imagCos = ImagCosOp::create(rewriter, loc, twoIm);

    Value denom = spirv::FAddOp::create(rewriter, loc, realCos, imagCos);
    Value resultRe = spirv::FDivOp::create(rewriter, loc, realSin, denom);
    Value resultIm = spirv::FDivOp::create(rewriter, loc, imagSin, denom);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

/// Computes complex.pow(lhs, rhs) following the polar decomposition
/// lhs^rhs = exp(c*log(abs(lhs)) - d*arg(lhs))
///         * (cos(c*arg(lhs) + d*log(abs(lhs)))
///            + i*sin(c*arg(lhs) + d*log(abs(lhs))))
/// where rhs = c + i*d.
template <typename SqrtOp, typename LogOp, typename ExpOp, typename SinOp,
          typename CosOp, typename Atan2Op, typename PowOp>
struct PowOpPattern final : OpConversionPattern<complex::PowOp> {
  using OpConversionPattern<complex::PowOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::PowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Value a = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {0});
    Value b = spirv::CompositeExtractOp::create(rewriter, loc, lhs, {1});
    Value c = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {0});
    Value d = spirv::CompositeExtractOp::create(rewriter, loc, rhs, {1});

    Value aSq = spirv::FMulOp::create(rewriter, loc, a, a);
    Value bSq = spirv::FMulOp::create(rewriter, loc, b, b);
    Value absSq = spirv::FAddOp::create(rewriter, loc, aSq, bSq);
    Value abs = SqrtOp::create(rewriter, loc, absSq);
    Value absToC = PowOp::create(rewriter, loc, abs, c);

    Value argLhs = Atan2Op::create(rewriter, loc, b, a);
    Value negD = spirv::FNegateOp::create(rewriter, loc, d);
    Value negDArgLhs = spirv::FMulOp::create(rewriter, loc, negD, argLhs);
    Value expNegDArgLhs = ExpOp::create(rewriter, loc, negDArgLhs);
    Value coeff = spirv::FMulOp::create(rewriter, loc, absToC, expNegDArgLhs);

    Value lnAbs = LogOp::create(rewriter, loc, abs);
    Value cArgLhs = spirv::FMulOp::create(rewriter, loc, c, argLhs);
    Value dLnAbs = spirv::FMulOp::create(rewriter, loc, d, lnAbs);
    Value q = spirv::FAddOp::create(rewriter, loc, cArgLhs, dLnAbs);
    Value cosQ = CosOp::create(rewriter, loc, q);
    Value sinQ = SinOp::create(rewriter, loc, q);

    Value resultRe = spirv::FMulOp::create(rewriter, loc, coeff, cosQ);
    Value resultIm = spirv::FMulOp::create(rewriter, loc, coeff, sinQ);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{resultRe, resultIm});
    return success();
  }
};

/// Computes complex.atan2(lhs, rhs) using the complex-logarithm identity
/// atan2(y, x) = -i * log((x + i*y) / sqrt(x**2 + y**2)), expressed directly
/// in terms of the real/imaginary components of x and y.
template <typename SqrtOp, typename SinOp, typename CosOp, typename Atan2Op,
          typename LogOp>
struct Atan2OpPattern final : OpConversionPattern<complex::Atan2Op> {
  using OpConversionPattern<complex::Atan2Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::Atan2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(op.getResult().getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    Location loc = op.getLoc();
    Value y = adaptor.getLhs();
    Value x = adaptor.getRhs();

    Value yRe = spirv::CompositeExtractOp::create(rewriter, loc, y, {0});
    Value yIm = spirv::CompositeExtractOp::create(rewriter, loc, y, {1});
    Value xRe = spirv::CompositeExtractOp::create(rewriter, loc, x, {0});
    Value xIm = spirv::CompositeExtractOp::create(rewriter, loc, x, {1});

    // w = x**2 + y**2.
    Value xReSq = spirv::FMulOp::create(rewriter, loc, xRe, xRe);
    Value xImSq = spirv::FMulOp::create(rewriter, loc, xIm, xIm);
    Value xSqRe = spirv::FSubOp::create(rewriter, loc, xReSq, xImSq);
    Value xReXIm = spirv::FMulOp::create(rewriter, loc, xRe, xIm);
    Value xSqIm = spirv::FAddOp::create(rewriter, loc, xReXIm, xReXIm);

    Value yReSq = spirv::FMulOp::create(rewriter, loc, yRe, yRe);
    Value yImSq = spirv::FMulOp::create(rewriter, loc, yIm, yIm);
    Value ySqRe = spirv::FSubOp::create(rewriter, loc, yReSq, yImSq);
    Value yReYIm = spirv::FMulOp::create(rewriter, loc, yRe, yIm);
    Value ySqIm = spirv::FAddOp::create(rewriter, loc, yReYIm, yReYIm);

    Value wRe = spirv::FAddOp::create(rewriter, loc, xSqRe, ySqRe);
    Value wIm = spirv::FAddOp::create(rewriter, loc, xSqIm, ySqIm);

    // sqrtW = sqrt(w), computed via the trigonometric form.
    auto [sqrtWRe, sqrtWIm] =
        createComplexSqrtOrRsqrt<SqrtOp, SinOp, CosOp, Atan2Op, SqrtOp>(
            rewriter, loc, wRe, wIm, /*angleScale=*/0.5);

    // num = x + i*y.
    Value numRe = spirv::FSubOp::create(rewriter, loc, xRe, yIm);
    Value numIm = spirv::FAddOp::create(rewriter, loc, xIm, yRe);

    // quotient = num / sqrtW.
    Value denomRe = spirv::FMulOp::create(rewriter, loc, sqrtWRe, sqrtWRe);
    Value denomIm = spirv::FMulOp::create(rewriter, loc, sqrtWIm, sqrtWIm);
    Value denom = spirv::FAddOp::create(rewriter, loc, denomRe, denomIm);

    Value numReSqrtWRe = spirv::FMulOp::create(rewriter, loc, numRe, sqrtWRe);
    Value numImSqrtWIm = spirv::FMulOp::create(rewriter, loc, numIm, sqrtWIm);
    Value quotNumRe =
        spirv::FAddOp::create(rewriter, loc, numReSqrtWRe, numImSqrtWIm);
    Value numImSqrtWRe = spirv::FMulOp::create(rewriter, loc, numIm, sqrtWRe);
    Value numReSqrtWIm = spirv::FMulOp::create(rewriter, loc, numRe, sqrtWIm);
    Value quotNumIm =
        spirv::FSubOp::create(rewriter, loc, numImSqrtWRe, numReSqrtWIm);

    Value quotRe = spirv::FDivOp::create(rewriter, loc, quotNumRe, denom);
    Value quotIm = spirv::FDivOp::create(rewriter, loc, quotNumIm, denom);

    // log(quotient) = log(abs(quotient)) + i*atan2(quotIm, quotRe).
    Value quotReSq = spirv::FMulOp::create(rewriter, loc, quotRe, quotRe);
    Value quotImSq = spirv::FMulOp::create(rewriter, loc, quotIm, quotIm);
    Value quotAbsSq = spirv::FAddOp::create(rewriter, loc, quotReSq, quotImSq);
    Value quotAbs = SqrtOp::create(rewriter, loc, quotAbsSq);
    Value logRe = LogOp::create(rewriter, loc, quotAbs);
    Value logIm = Atan2Op::create(rewriter, loc, quotIm, quotRe);

    // result = -i * log(quotient) = logIm - i*logRe.
    Value resultIm = spirv::FNegateOp::create(rewriter, loc, logRe);

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        op, spirvType, llvm::ArrayRef<Value>{logIm, resultIm});
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateComplexToSPIRVPatterns(
    const SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns
      .add<ConstantOpPattern, CreateOpPattern, ReOpPattern, ImOpPattern,
           ElementwiseBinaryOpPattern<complex::AddOp, spirv::FAddOp>,
           ElementwiseBinaryOpPattern<complex::SubOp, spirv::FSubOp>,
           ComparisonOpPattern<complex::EqualOp, spirv::FOrdEqualOp,
                               spirv::LogicalAndOp>,
           ComparisonOpPattern<complex::NotEqualOp, spirv::FUnordNotEqualOp,
                               spirv::LogicalOrOp>,
           MulOpPattern, DivOpPattern,
           NegationOpPattern<complex::NegOp, /*NegateReal=*/true>,
           NegationOpPattern<complex::ConjOp, /*NegateReal=*/false>,
           AbsOpPattern<spirv::GLSqrtOp>, AbsOpPattern<spirv::CLSqrtOp>,
           AngleOpPattern<spirv::GLAtan2Op>, AngleOpPattern<spirv::CLAtan2Op>,
           SignOpPattern<spirv::GLSqrtOp>, SignOpPattern<spirv::CLSqrtOp>,
           SinOpPattern<spirv::GLSinOp, spirv::GLCosOp, spirv::GLExpOp>,
           SinOpPattern<spirv::CLSinOp, spirv::CLCosOp, spirv::CLExpOp>,
           CosOpPattern<spirv::GLSinOp, spirv::GLCosOp, spirv::GLExpOp>,
           CosOpPattern<spirv::CLSinOp, spirv::CLCosOp, spirv::CLExpOp>,
           ExpOpPattern<spirv::GLExpOp, spirv::GLSinOp, spirv::GLCosOp>,
           ExpOpPattern<spirv::CLExpOp, spirv::CLSinOp, spirv::CLCosOp>,
           Expm1OpPattern<spirv::GLExpOp, spirv::GLSinOp, spirv::GLCosOp>,
           Expm1OpPattern<spirv::CLExpOp, spirv::CLSinOp, spirv::CLCosOp>,
           LogOpPattern<spirv::GLSqrtOp, spirv::GLLogOp, spirv::GLAtan2Op>,
           LogOpPattern<spirv::CLSqrtOp, spirv::CLLogOp, spirv::CLAtan2Op>,
           Log1pOpPattern<spirv::GLSqrtOp, spirv::GLLogOp, spirv::GLAtan2Op>,
           Log1pOpPattern<spirv::CLSqrtOp, spirv::CLLogOp, spirv::CLAtan2Op>,
           SqrtOpPattern<spirv::GLSqrtOp, spirv::GLSinOp, spirv::GLCosOp,
                         spirv::GLAtan2Op>,
           SqrtOpPattern<spirv::CLSqrtOp, spirv::CLSinOp, spirv::CLCosOp,
                         spirv::CLAtan2Op>,
           RsqrtOpPattern<spirv::GLSqrtOp, spirv::GLSinOp, spirv::GLCosOp,
                          spirv::GLAtan2Op, spirv::GLInverseSqrtOp>,
           RsqrtOpPattern<spirv::CLSqrtOp, spirv::CLSinOp, spirv::CLCosOp,
                          spirv::CLAtan2Op, spirv::CLRsqrtOp>,
           TanTanhOpPattern<complex::TanOp, spirv::GLSinOp, spirv::GLCosOp,
                            spirv::GLSinhOp, spirv::GLCoshOp>,
           TanTanhOpPattern<complex::TanOp, spirv::CLSinOp, spirv::CLCosOp,
                            spirv::CLSinhOp, spirv::CLCoshOp>,
           TanTanhOpPattern<complex::TanhOp, spirv::GLSinhOp, spirv::GLCoshOp,
                            spirv::GLSinOp, spirv::GLCosOp>,
           TanTanhOpPattern<complex::TanhOp, spirv::CLSinhOp, spirv::CLCoshOp,
                            spirv::CLSinOp, spirv::CLCosOp>,
           PowOpPattern<spirv::GLSqrtOp, spirv::GLLogOp, spirv::GLExpOp,
                        spirv::GLSinOp, spirv::GLCosOp, spirv::GLAtan2Op,
                        spirv::GLPowOp>,
           PowOpPattern<spirv::CLSqrtOp, spirv::CLLogOp, spirv::CLExpOp,
                        spirv::CLSinOp, spirv::CLCosOp, spirv::CLAtan2Op,
                        spirv::CLPowOp>,
           Atan2OpPattern<spirv::GLSqrtOp, spirv::GLSinOp, spirv::GLCosOp,
                          spirv::GLAtan2Op, spirv::GLLogOp>,
           Atan2OpPattern<spirv::CLSqrtOp, spirv::CLSinOp, spirv::CLCosOp,
                          spirv::CLAtan2Op, spirv::CLLogOp>>(typeConverter,
                                                             context);
}
