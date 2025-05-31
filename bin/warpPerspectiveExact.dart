import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:isolate';
import 'dart:async';
import 'dart:io';
import 'dart:convert';

const int INTER_BITS = 5;
const int INTER_TAB_SIZE = 1 << INTER_BITS; // 32
const int INTER_REMAP_COEF_BITS = 15;
const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS; // 32768

const int CV_INTER_CUBIC = 2;
const int CV_INTER_LANCZOS4 = 4;
const int CV_BORDER_REPLICATE = 1;
const int CV_WARP_INVERSE_MAP = 16;

const double FLT_EPSILON_FLOAT32 = 1.1920928955078125e-7;

class _CacheKey {
  final int interpolationMethod;
  final bool forFixpoint;

  _CacheKey(this.interpolationMethod, this.forFixpoint);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
          other is _CacheKey &&
              runtimeType == other.runtimeType &&
              interpolationMethod == other.interpolationMethod &&
              forFixpoint == other.forFixpoint;

  @override
  int get hashCode => interpolationMethod.hashCode ^ forFixpoint.hashCode;
}

final Map<_CacheKey, dynamic> PRECOMPUTED_TABLES_CACHE = {};

double _Round(double value) {
  if (!value.isFinite) {
    return value;
  }
  if ((value - value.floor() - 0.5).abs() < 1e-9) {
    return value.floor().isEven ? value.floorToDouble() : value.ceilToDouble();
  } else if ((value - value.ceil() + 0.5).abs() < 1e-9) {
    return value.ceil().isEven ? value.ceilToDouble() : value.floorToDouble();
  }
  return value.roundToDouble();
}

int saturateCastUint8(double v) {
  return _Round(v).clamp(0, 255).toInt();
}

int saturateCastInt16(double v) {
  return _Round(v).clamp(-32768, 32767).toInt();
}

int clipInt(int val, int minVal, int maxVal) {
  return math.max(minVal, math.min(val, maxVal));
}

void _interpolateCubicCoeffs1D(double xFloat, Float64List coeffsArrOut) {
  double x = xFloat;
  const double A = -0.75;
  coeffsArrOut[0] = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
  coeffsArrOut[1] = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
  coeffsArrOut[2] = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
  coeffsArrOut[3] = 1.0 - (coeffsArrOut[0] + coeffsArrOut[1] + coeffsArrOut[2]);
}

void _interpolateLanczos4Coeffs1D(double xFloat, Float64List coeffsArrOut) {
  const double S45 = 0.70710678118654752440084436210485;
  final List<List<double>> CS_TABLE = [
    [1.0, 0.0], [-S45, -S45], [0.0, 1.0], [S45, -S45],
    [-1.0, 0.0], [S45, S45], [0.0, -1.0], [-S45, S45]
  ];

  double x = xFloat;

  if (x < FLT_EPSILON_FLOAT32) {
    for (int i = 0; i < coeffsArrOut.length; i++) coeffsArrOut[i] = 0.0;
    coeffsArrOut[3] = 1.0;
    return;
  }

  double currentSum = 0.0;
  double y0 = -(x + 3.0) * math.pi * 0.25;
  double s0 = math.sin(y0);
  double c0 = math.cos(y0);

  Float64List tempCoeffsF64 = Float64List(8);

  for (int i = 0; i < 8; i++) {
    double y = -(x + 3.0 - i) * math.pi * 0.25;
    double numerator = CS_TABLE[i][0] * s0 + CS_TABLE[i][1] * c0;
    double denominator = y * y;
    tempCoeffsF64[i] = numerator / denominator;
    currentSum += tempCoeffsF64[i];
  }

  if (currentSum.abs() < 1e-9) {
    for (int i = 0; i < coeffsArrOut.length; i++) coeffsArrOut[i] = 0.0;
    coeffsArrOut[3] = 1.0;
  } else {
    double invSum = 1.0 / currentSum;
    for (int i = 0; i < 8; i++) {
      coeffsArrOut[i] = tempCoeffsF64[i] * invSum;
    }
  }
}

// --- Logic khởi tạo bảng ---
void _initCoeffs1DAllAlpha(int method, List<Float32List> coeffs1DStorageF32) {
  int kSize = coeffs1DStorageF32[0].length;
  double scale = 1.0 / INTER_TAB_SIZE;
  Float64List tempCoeffsForOneAlphaF64 = Float64List(kSize);

  for (int i = 0; i < INTER_TAB_SIZE; i++) {
    double xFrac = i * scale;
    if (method == CV_INTER_CUBIC) {
      _interpolateCubicCoeffs1D(xFrac, tempCoeffsForOneAlphaF64);
    } else if (method == CV_INTER_LANCZOS4) {
      _interpolateLanczos4Coeffs1D(xFrac, tempCoeffsForOneAlphaF64);
    } else {
      throw ArgumentError("Phương thức nội suy không được hỗ trợ cho bảng 1D");
    }
    for (int k = 0; k < kSize; k++) {
      coeffs1DStorageF32[i][k] = tempCoeffsForOneAlphaF64[k];
    }
  }
}

dynamic _getOrComputeInterpolationTable(int interpolationMethod, bool forFixpoint) {
  _CacheKey cacheKey = _CacheKey(interpolationMethod, forFixpoint);
  if (PRECOMPUTED_TABLES_CACHE.containsKey(cacheKey)) {
    return PRECOMPUTED_TABLES_CACHE[cacheKey]!;
  }

  int kSize;
  if (interpolationMethod == CV_INTER_CUBIC) {
    kSize = 4;
  } else if (interpolationMethod == CV_INTER_LANCZOS4) {
    kSize = 8;
  } else {
    throw ArgumentError("Phương thức nội suy không được hỗ trợ để tạo bảng.");
  }

  List<Float32List> coeffs1DAllAlphasF32 = List.generate(
      INTER_TAB_SIZE, (_) => Float32List(kSize), growable: false
  );
  _initCoeffs1DAllAlpha(interpolationMethod, coeffs1DAllAlphasF32);

  int numAlphaCombinations = INTER_TAB_SIZE * INTER_TAB_SIZE;

  dynamic wtab;
  int wtabEntrySize = kSize * kSize;

  if (forFixpoint) {
    wtab = Int16List(numAlphaCombinations * wtabEntrySize);
  } else {
    wtab = Float32List(numAlphaCombinations * wtabEntrySize);
  }

  Float64List coeffsY1D = Float64List(kSize);
  Float64List coeffsX1D = Float64List(kSize);
  Float64List weights2DFlatF64 = Float64List(wtabEntrySize);
  Int16List intWeightsCurrentEntry = Int16List(wtabEntrySize);

  for (int ayIdx = 0; ayIdx < INTER_TAB_SIZE; ayIdx++) {
    for(int k=0; k<kSize; ++k) coeffsY1D[k] = coeffs1DAllAlphasF32[ayIdx][k];

    for (int axIdx = 0; axIdx < INTER_TAB_SIZE; axIdx++) {
      for(int k=0; k<kSize; ++k) coeffsX1D[k] = coeffs1DAllAlphasF32[axIdx][k];

      int wtabEntryIdx = ayIdx * INTER_TAB_SIZE + axIdx;

      for (int r = 0; r < kSize; r++) {
        for (int c = 0; c < kSize; c++) {
          weights2DFlatF64[r * kSize + c] = coeffsY1D[r] * coeffsX1D[c];
        }
      }

      if (forFixpoint) {
        double scaleFactor = INTER_REMAP_COEF_SCALE.toDouble();
        for (int k = 0; k < wtabEntrySize; k++) {
          intWeightsCurrentEntry[k] = saturateCastInt16(weights2DFlatF64[k] * scaleFactor);
        }

        int currentSumOfIntWeights = 0;
        for (int k = 0; k < wtabEntrySize; k++) {
          currentSumOfIntWeights += intWeightsCurrentEntry[k];
        }

        int targetSumInt64 = INTER_REMAP_COEF_SCALE;

        if (currentSumOfIntWeights != targetSumInt64) {
          int diff = currentSumOfIntWeights - targetSumInt64;

          int kHalf = kSize ~/ 2;
          int minValInCenter = 32767 + 1;
          int maxValInCenter = -32768 - 1;
          int minValFlatIdx = -1;
          int maxValFlatIdx = -1;

          for (int rKernelCenter = kHalf; rKernelCenter < kHalf + 2; rKernelCenter++) {
            for (int cKernelCenter = kHalf; cKernelCenter < kHalf + 2; cKernelCenter++) {
              int flatIdx = rKernelCenter * kSize + cKernelCenter;
              if (flatIdx < wtabEntrySize) {
                int valAtIdx = intWeightsCurrentEntry[flatIdx];
                if (valAtIdx < minValInCenter) {
                  minValInCenter = valAtIdx;
                  minValFlatIdx = flatIdx;
                }
                if (valAtIdx > maxValInCenter) {
                  maxValInCenter = valAtIdx;
                  maxValFlatIdx = flatIdx;
                }
              }
            }
          }

          if (diff < 0) {
            if (maxValFlatIdx != -1) {
              double valToAdjust = intWeightsCurrentEntry[maxValFlatIdx].toDouble() - diff;
              intWeightsCurrentEntry[maxValFlatIdx] = saturateCastInt16(valToAdjust);
            }
          } else {
            if (minValFlatIdx != -1) {
              double valToAdjust = intWeightsCurrentEntry[minValFlatIdx].toDouble() - diff;
              intWeightsCurrentEntry[minValFlatIdx] = saturateCastInt16(valToAdjust);
            }
          }
        }
        int offset = wtabEntryIdx * wtabEntrySize;
        for(int k=0; k<wtabEntrySize; ++k) (wtab as Int16List)[offset + k] = intWeightsCurrentEntry[k];

      } else {
        int offset = wtabEntryIdx * wtabEntrySize;
        for(int k=0; k<wtabEntrySize; ++k) (wtab as Float32List)[offset + k] = weights2DFlatF64[k];
      }
    }
  }

  PRECOMPUTED_TABLES_CACHE[cacheKey] = wtab;
  return wtab;
}

class _RemapIsolateData {
  final SendPort sendPort;
  final Uint8List srcImageData;
  final int srcRows;
  final int srcCols;
  final int numChannels;
  final bool isSrcUint8;
  final Float32List mapXCoordsFloat;
  final Float32List mapYCoordsFloat;
  final int dstCols;
  final dynamic wtabData;
  final int wtabEntrySize;
  final int kSize;
  final int kernelCoordOffset;
  final int startDstRow;
  final int endDstRow;
  final int totalDstRows;

  _RemapIsolateData({
    required this.sendPort,
    required this.srcImageData,
    required this.srcRows,
    required this.srcCols,
    required this.numChannels,
    required this.isSrcUint8,
    required this.mapXCoordsFloat,
    required this.mapYCoordsFloat,
    required this.dstCols,
    required this.wtabData,
    required this.wtabEntrySize,
    required this.kSize,
    required this.kernelCoordOffset,
    required this.startDstRow,
    required this.endDstRow,
    required this.totalDstRows,
  });
}

void _remapIsolateWorker(_RemapIsolateData data) {
  final Uint8List srcImageData = data.srcImageData;
  final Float32List mapXCoordsFloat = data.mapXCoordsFloat;
  final Float32List mapYCoordsFloat = data.mapYCoordsFloat;
  final dynamic typedWtab = data.wtabData;

  final int srcRows = data.srcRows;
  final int srcCols = data.srcCols;
  final int numChannels = data.numChannels;
  final bool isSrcUint8 = data.isSrcUint8;
  final int dstCols = data.dstCols;
  final int wtabEntrySize = data.wtabEntrySize;
  final int kSize = data.kSize;
  final int kernelCoordOffset = data.kernelCoordOffset;
  final int startDstRow = data.startDstRow;
  final int endDstRow = data.endDstRow;

  final int rowsToProcess = endDstRow - startDstRow;
  dynamic dstChunkData;
  if (isSrcUint8) {
    dstChunkData = Uint8List(rowsToProcess * dstCols * numChannels);
  } else {
    dstChunkData = Float32List(rowsToProcess * dstCols * numChannels);
  }

  Int32List ixAll = Int32List(dstCols);
  Int32List iyAll = Int32List(dstCols);
  Int32List mapIntXBase = Int32List(dstCols);
  Int32List mapIntYBase = Int32List(dstCols);
  Int32List alphaXIndices = Int32List(dstCols);
  Int32List alphaYIndices = Int32List(dstCols);
  Int32List mapFracTableIndices = Int32List(dstCols);

  List<int> pixelSumAccInt = isSrcUint8 ? List.filled(numChannels, 0) : [];
  List<double> pixelSumAccFloat = !isSrcUint8 ? List.filled(numChannels, 0.0) : [];

  for (int rDst = startDstRow; rDst < endDstRow; rDst++) {
    for (int cDst = 0; cDst < dstCols; cDst++) {
      int mapIdx = rDst * dstCols + cDst;
      double scaledMapX = mapXCoordsFloat[mapIdx] * INTER_TAB_SIZE;
      double scaledMapY = mapYCoordsFloat[mapIdx] * INTER_TAB_SIZE;

      ixAll[cDst] = _Round(scaledMapX).toInt();
      iyAll[cDst] = _Round(scaledMapY).toInt();

      mapIntXBase[cDst] = ixAll[cDst] >> INTER_BITS;
      mapIntYBase[cDst] = iyAll[cDst] >> INTER_BITS;

      alphaXIndices[cDst] = ixAll[cDst] & (INTER_TAB_SIZE - 1);
      alphaYIndices[cDst] = iyAll[cDst] & (INTER_TAB_SIZE - 1);
      mapFracTableIndices[cDst] = alphaYIndices[cDst] * INTER_TAB_SIZE + alphaXIndices[cDst];
    }
    for (int cDst = 0; cDst < dstCols; cDst++) {
      int baseSrcXForKernel = mapIntXBase[cDst] - kernelCoordOffset;
      int baseSrcYForKernel = mapIntYBase[cDst] - kernelCoordOffset;

      int fracIdxForWtab = mapFracTableIndices[cDst];
      int wtabOffset = fracIdxForWtab * wtabEntrySize;

      if (isSrcUint8) {
        for (int ch = 0; ch < numChannels; ch++) pixelSumAccInt[ch] = 0;
      } else {
        for (int ch = 0; ch < numChannels; ch++) pixelSumAccFloat[ch] = 0.0;
      }

      for (int rKernel = 0; rKernel < kSize; rKernel++) {
        for (int cKernel = 0; cKernel < kSize; cKernel++) {
          int currentSrcYMap = baseSrcYForKernel + rKernel;
          int currentSrcXMap = baseSrcXForKernel + cKernel;

          int readSrcY = clipInt(currentSrcYMap, 0, srcRows - 1);
          int readSrcX = clipInt(currentSrcXMap, 0, srcCols - 1);

          int srcPixelOffset = (readSrcY * srcCols + readSrcX) * numChannels;

          dynamic weightDyn = (typedWtab is Int16List)
              ? typedWtab[wtabOffset + rKernel * kSize + cKernel]
              : (typedWtab as Float32List)[wtabOffset + rKernel * kSize + cKernel];

          if (isSrcUint8) {
            int weight = weightDyn as int;
            for (int ch = 0; ch < numChannels; ch++) {
              int srcPixelVal = srcImageData[srcPixelOffset + ch];
              pixelSumAccInt[ch] += srcPixelVal * weight;
            }
          } else {
            double weight = weightDyn as double;
            for (int ch = 0; ch < numChannels; ch++) {
              double srcPixelVal = (srcImageData as dynamic)[srcPixelOffset + ch].toDouble();
              pixelSumAccFloat[ch] += srcPixelVal * weight;
            }
          }
        }
      }
      int dstPixelOffsetInChunk = ((rDst - startDstRow) * dstCols + cDst) * numChannels;

      if (isSrcUint8) {
        const int delta = (1 << (INTER_REMAP_COEF_BITS - 1));
        for (int ch = 0; ch < numChannels; ch++) {
          int val = (pixelSumAccInt[ch] + delta) >> INTER_REMAP_COEF_BITS;
          (dstChunkData as Uint8List)[dstPixelOffsetInChunk + ch] = saturateCastUint8(val.toDouble());
        }
      } else {
        for (int ch = 0; ch < numChannels; ch++) {
          (dstChunkData as Float32List)[dstPixelOffsetInChunk + ch] = pixelSumAccFloat[ch];
        }
      }
    }
  }
  if (dstChunkData is Uint8List) {
    data.sendPort.send(TransferableTypedData.fromList([dstChunkData]));
  } else if (dstChunkData is Float32List) {
    data.sendPort.send(TransferableTypedData.fromList([dstChunkData]));
  } else {
    data.sendPort.send(dstChunkData);
  }
}
class _WarpIsolateResult {
  final int startRow;
  final Uint8List chunkData;

  _WarpIsolateResult(this.startRow, this.chunkData);
}

Future<Map<String, dynamic>> warpPerspectiveExact({
  required Uint8List srcImageHWC,
  required int srcHeight,
  required int srcWidth,
  int? srcNumChannels,
  required List<List<double>> M_list,
  required List<int> dsize,
  int flags = CV_INTER_CUBIC,
  int borderMode = CV_BORDER_REPLICATE,
  int numIsolates = 0,
}) async {
  if (borderMode != CV_BORDER_REPLICATE) {
    throw ArgumentError("Chỉ hỗ trợ CV_BORDER_REPLICATE cho borderMode.");
  }

  int actualNumChannels = srcNumChannels ?? (srcImageHWC.length ~/ (srcHeight * srcWidth));
  if (srcImageHWC.length != srcHeight * srcWidth * actualNumChannels) {
    throw ArgumentError("Kích thước srcImageHWC không khớp với srcHeight, srcWidth, srcNumChannels.");
  }

  Matrix3x3 M = Matrix3x3(M_list);

  if (!((flags & CV_WARP_INVERSE_MAP) != 0)) {
    M = M.inverse();
  }

  final int dstWidth = dsize[0];
  final int dstHeight = dsize[1];

  // Tạo lưới tọa độ cho ảnh đích
  Float64List xCoordsFlat = Float64List(dstHeight * dstWidth);
  Float64List yCoordsFlat = Float64List(dstHeight * dstWidth);
  Float64List onesFlat = Float64List(dstHeight * dstWidth);

  for (int r = 0; r < dstHeight; ++r) {
    for (int c = 0; c < dstWidth; ++c) {
      int idx = r * dstWidth + c;
      xCoordsFlat[idx] = c.toDouble();
      yCoordsFlat[idx] = r.toDouble();
      onesFlat[idx] = 1.0;
    }
  }

  // Áp dụng biến đổi perspective
  List<Float64List> transformedCoords = M.multiplyByCoords(xCoordsFlat, yCoordsFlat, onesFlat);
  Float64List transformedX = transformedCoords[0];
  Float64List transformedY = transformedCoords[1];
  Float64List transformedW = transformedCoords[2];

  Float32List mapX = Float32List(dstHeight * dstWidth);
  Float32List mapY = Float32List(dstHeight * dstWidth);

  for (int i = 0; i < transformedW.length; ++i) {
    double w = transformedW[i];
    if (w.abs() > 1e-10) {
      mapX[i] = (transformedX[i] / w);
      mapY[i] = (transformedY[i] / w);
    } else {
      mapX[i] = 0.0;
      mapY[i] = 0.0;
    }
  }

  const bool IS_SRC_UINT8 = true;
  int INTERPOLATION_METHOD = flags;

  dynamic wtab = _getOrComputeInterpolationTable(INTERPOLATION_METHOD, IS_SRC_UINT8);
  final int wtabEntrySize = (wtab is Int16List ? wtab.length : (wtab as Float32List).length) ~/ (INTER_TAB_SIZE * INTER_TAB_SIZE);

  int kSize, kernelCoordOffset;
  if (INTERPOLATION_METHOD == CV_INTER_CUBIC) {
    kSize = 4;
    kernelCoordOffset = 1;
  } else {
    kSize = 8;
    kernelCoordOffset = 3;
  }
  Uint8List dstImage = Uint8List(dstHeight * dstWidth * actualNumChannels);

  int actualNumIsolates = numIsolates;
  if (actualNumIsolates <= 0) {
    actualNumIsolates = math.max(1, Platform.numberOfProcessors);
  }
  if (dstHeight == 0 || dstWidth == 0) {
    return {
      'data': dstImage,
      'width': dstWidth,
      'height': dstHeight,
      'channels': actualNumChannels,
      'dtype': Uint8List,
    };
  }
  actualNumIsolates = math.min(actualNumIsolates, dstHeight);

  if (actualNumIsolates == 1) {
    print("Chạy warpPerspective với 1 luồng.");
    Uint8List singleThreadResult = _executeRemapLogicInline(
        srcImageHWC, srcHeight, srcWidth, actualNumChannels, IS_SRC_UINT8,
        mapX, mapY, dstWidth, dstHeight,
        wtab, wtabEntrySize, kSize, kernelCoordOffset
    );
    dstImage.setAll(0, singleThreadResult);
  } else {
    print("Chạy warpPerspective với $actualNumIsolates luồng.");
    final List<Future<_WarpIsolateResult>> isolateFutures = [];

    int rowsPerIsolate = dstHeight ~/ actualNumIsolates;
    int remainingRows = dstHeight % actualNumIsolates;

    for (int i = 0; i < actualNumIsolates; i++) {
      int startRow = i * rowsPerIsolate + math.min(i, remainingRows);
      int endRow = startRow + rowsPerIsolate + (i < remainingRows ? 1 : 0);
      if (startRow >= endRow) continue;

      final Completer<_WarpIsolateResult> completer = Completer();
      isolateFutures.add(completer.future);

      final ReceivePort isolateResponsePort = ReceivePort();
      isolateResponsePort.listen((message) {
        if (message is TransferableTypedData) {
          completer.complete(_WarpIsolateResult(startRow, message.materialize().asUint8List()));
        } else if (message is Uint8List) {
          completer.complete(_WarpIsolateResult(startRow, message));
        } else {
          completer.completeError("Loại tin nhắn không mong muốn từ isolate: ${message.runtimeType}");
        }
        isolateResponsePort.close();
      });

      Isolate.spawn(
        _remapIsolateWorker,
        _RemapIsolateData(
          sendPort: isolateResponsePort.sendPort,
          srcImageData: srcImageHWC,
          srcRows: srcHeight,
          srcCols: srcWidth,
          numChannels: actualNumChannels,
          isSrcUint8: IS_SRC_UINT8,
          mapXCoordsFloat: mapX,
          mapYCoordsFloat: mapY,
          dstCols: dstWidth,
          wtabData: wtab,
          wtabEntrySize: wtabEntrySize,
          kSize: kSize,
          kernelCoordOffset: kernelCoordOffset,
          startDstRow: startRow,
          endDstRow: endRow,
          totalDstRows: dstHeight,
        ),
      );
    }
    final List<_WarpIsolateResult> results = await Future.wait(isolateFutures);
    for (final result in results) {
      final Uint8List chunk = result.chunkData;
      int chunkDstOffset = result.startRow * dstWidth * actualNumChannels;
      int endIndex = chunkDstOffset + chunk.length;

      if (endIndex <= dstImage.length && chunkDstOffset >= 0) {
        dstImage.setRange(chunkDstOffset, endIndex, chunk);
      } else {
        print("CẢNH BÁO: Lỗi chỉ số khi ghép chunk. startRow: ${result.startRow}, chunkLength: ${chunk.length}, dstImageLength: ${dstImage.length}");
      }
    }
  }

  return {
    'data': dstImage,
    'width': dstWidth,
    'height': dstHeight,
    'channels': actualNumChannels,
    'dtype': Uint8List,
  };
}

Uint8List _executeRemapLogicInline(
    Uint8List srcImageData, int srcRows, int srcCols, int numChannels, bool isSrcUint8,
    Float32List mapXCoordsFloat, Float32List mapYCoordsFloat, int dstCols, int totalDstRows,
    dynamic wtab, int wtabEntrySize, int kSize, int kernelCoordOffset
    ) {
  Uint8List dstResult = Uint8List(totalDstRows * dstCols * numChannels);

  Int32List ixAll = Int32List(dstCols);
  Int32List iyAll = Int32List(dstCols);
  Int32List mapIntXBase = Int32List(dstCols);
  Int32List mapIntYBase = Int32List(dstCols);
  Int32List alphaXIndices = Int32List(dstCols);
  Int32List alphaYIndices = Int32List(dstCols);
  Int32List mapFracTableIndices = Int32List(dstCols);

  List<int> pixelSumAccInt = isSrcUint8 ? List.filled(numChannels, 0) : [];

  for (int rDst = 0; rDst < totalDstRows; rDst++) {
    for (int cDst = 0; cDst < dstCols; cDst++) {
      int mapIdx = rDst * dstCols + cDst;
      double scaledMapX = mapXCoordsFloat[mapIdx] * INTER_TAB_SIZE;
      double scaledMapY = mapYCoordsFloat[mapIdx] * INTER_TAB_SIZE;

      ixAll[cDst] = _Round(scaledMapX).toInt();
      iyAll[cDst] = _Round(scaledMapY).toInt();

      mapIntXBase[cDst] = ixAll[cDst] >> INTER_BITS;
      mapIntYBase[cDst] = iyAll[cDst] >> INTER_BITS;

      alphaXIndices[cDst] = ixAll[cDst] & (INTER_TAB_SIZE - 1);
      alphaYIndices[cDst] = iyAll[cDst] & (INTER_TAB_SIZE - 1);
      mapFracTableIndices[cDst] = alphaYIndices[cDst] * INTER_TAB_SIZE + alphaXIndices[cDst];
    }

    for (int cDst = 0; cDst < dstCols; cDst++) {
      int baseSrcXForKernel = mapIntXBase[cDst] - kernelCoordOffset;
      int baseSrcYForKernel = mapIntYBase[cDst] - kernelCoordOffset;

      int fracIdxForWtab = mapFracTableIndices[cDst];
      int wtabOffset = fracIdxForWtab * wtabEntrySize;

      for (int ch = 0; ch < numChannels; ch++) pixelSumAccInt[ch] = 0;

      for (int rKernel = 0; rKernel < kSize; rKernel++) {
        for (int cKernel = 0; cKernel < kSize; cKernel++) {
          int currentSrcYMap = baseSrcYForKernel + rKernel;
          int currentSrcXMap = baseSrcXForKernel + cKernel;

          int readSrcY = clipInt(currentSrcYMap, 0, srcRows - 1);
          int readSrcX = clipInt(currentSrcXMap, 0, srcCols - 1);

          int srcPixelOffset = (readSrcY * srcCols + readSrcX) * numChannels;

          int weight = (wtab as Int16List)[wtabOffset + rKernel * kSize + cKernel];

          for (int ch = 0; ch < numChannels; ch++) {
            int srcPixelVal = srcImageData[srcPixelOffset + ch];
            pixelSumAccInt[ch] += srcPixelVal * weight;
          }
        }
      }

      int dstPixelOffset = (rDst * dstCols + cDst) * numChannels;
      const int delta = (1 << (INTER_REMAP_COEF_BITS - 1));
      for (int ch = 0; ch < numChannels; ch++) {
        int val = (pixelSumAccInt[ch] + delta) >> INTER_REMAP_COEF_BITS;
        dstResult[dstPixelOffset + ch] = saturateCastUint8(val.toDouble());
      }
    }
  }
  return dstResult;
}

class Matrix3x3 {
  final Float64List _data;

  Matrix3x3(List<List<double>> M) : _data = Float64List(9) {
    assert(M.length == 3 && M.every((row) => row.length == 3));
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        _data[i * 3 + j] = M[i][j];
      }
    }
  }

  Matrix3x3.fromFlatList(List<double> data) : _data = Float64List.fromList(data) {
    assert(data.length == 9);
  }

  double get(int r, int c) => _data[r * 3 + c];

  Matrix3x3 inverse() {
    double det = get(0,0)*(get(1,1)*get(2,2) - get(2,1)*get(1,2)) -
        get(0,1)*(get(1,0)*get(2,2) - get(1,2)*get(2,0)) +
        get(0,2)*(get(1,0)*get(2,1) - get(1,1)*get(2,0));

    if (det.abs() < 1e-18) {
      throw Exception("Ma trận suy biến và không thể nghịch đảo.");
    }

    double invDet = 1.0 / det;
    Float64List invData = Float64List(9);

    invData[0] = (get(1,1)*get(2,2) - get(2,1)*get(1,2)) * invDet;
    invData[1] = (get(0,2)*get(2,1) - get(0,1)*get(2,2)) * invDet;
    invData[2] = (get(0,1)*get(1,2) - get(0,2)*get(1,1)) * invDet;
    invData[3] = (get(1,2)*get(2,0) - get(1,0)*get(2,2)) * invDet;
    invData[4] = (get(0,0)*get(2,2) - get(0,2)*get(2,0)) * invDet;
    invData[5] = (get(1,0)*get(0,2) - get(0,0)*get(1,2)) * invDet;
    invData[6] = (get(1,0)*get(2,1) - get(2,0)*get(1,1)) * invDet;
    invData[7] = (get(2,0)*get(0,1) - get(0,0)*get(2,1)) * invDet;
    invData[8] = (get(0,0)*get(1,1) - get(1,0)*get(0,1)) * invDet;

    return Matrix3x3.fromFlatList(invData);
  }

  List<Float64List> multiplyByCoords(Float64List xCoords, Float64List yCoords, Float64List ones) {
    int N = xCoords.length;
    assert(yCoords.length == N && ones.length == N);

    Float64List tx = Float64List(N);
    Float64List ty = Float64List(N);
    Float64List tw = Float64List(N);

    for (int i = 0; i < N; ++i) {
      double x = xCoords[i];
      double y = yCoords[i];
      double wVal = ones[i];

      tx[i] = get(0,0)*x + get(0,1)*y + get(0,2)*wVal;
      ty[i] = get(1,0)*x + get(1,1)*y + get(1,2)*wVal;
      tw[i] = get(2,0)*x + get(2,1)*y + get(2,2)*wVal;
    }
    return [tx, ty, tw];
  }
}


/// ======================= RUN TEST =============================

void compareResultsDart(
    Uint8List resultDart,
    int height,
    int width,
    int channels,
    List<dynamic> opencvResultDataFlat,
    String testName) {
  if (resultDart.length != opencvResultDataFlat.length) {
    print("$testName: LỖI KÍCH THƯỚC! "
        "Dart: ${resultDart.length}, OpenCV (từ JSON): ${opencvResultDataFlat.length}");
    print("  Dart shape (ước tính): $height x $width x $channels");
    return;
  }

  List<double> diffList = [];
  int pixelsDiffGt1 = 0;

  for (int i = 0; i < resultDart.length; i++) {
    double diff = (resultDart[i] - (opencvResultDataFlat[i] as int)).abs().toDouble();
    diffList.add(diff);
    if (diff > 1.0) {
      pixelsDiffGt1++;
    }
  }

  if (diffList.isEmpty) {
    print("$testName: Không có dữ liệu để so sánh.");
    return;
  }

  double maxDiff = diffList.reduce(math.max);
  double sumDiff = diffList.reduce((a, b) => a + b);
  double meanDiff = sumDiff / diffList.length;

  print("$testName:");
  print("  Sai khác tối đa: $maxDiff");
  print("  Sai khác trung bình: ${meanDiff.toStringAsFixed(4)}");
  print("  Số pixel có sai khác > 1: $pixelsDiffGt1/${diffList.length} "
      "(${(100.0 * pixelsDiffGt1 / diffList.length).toStringAsFixed(2)}%)");
  print("");

  if (maxDiff > 1.0) {
    print("CẢNH BÁO: Sai khác tối đa > 1.0. Cần kiểm tra kỹ lưỡng!");
    for (int i = 0; i < resultDart.length; i++) {
      if ((resultDart[i] - (opencvResultDataFlat[i] as int)).abs() > 1) {
        int pixelIndex = i ~/ channels;
        int r = pixelIndex ~/ width;
        int c = pixelIndex % width;
        int ch = i % channels;
        print("  Diff at (r:$r, c:$c, ch:$ch): Dart=${resultDart[i]}, OpenCV=${opencvResultDataFlat[i]}");
      }
    }
  } else {
    print("THÀNH CÔNG: Kết quả Dart khớp với OpenCV (sai khác tối đa <= 1.0).");
  }
}

Future<void> main(List<String> arguments) async {
  final String jsonFilePath = arguments.isNotEmpty ? arguments[0] : 'test_warp_data.json';
  final File jsonFile = File(jsonFilePath);

  if (!await jsonFile.exists()) {
    print("Lỗi: File JSON '$jsonFilePath' không tồn tại. "
        "Hãy chạy script Python để tạo file này trước.");
    return;
  }
  print("Đọc dữ liệu từ file JSON: $jsonFilePath");
  final String jsonString = await jsonFile.readAsString();
  final Map<String, dynamic> testData = json.decode(jsonString);

  final Map<String, dynamic> inputImageDataJson = testData['input_image'];
  final int srcHeight = inputImageDataJson['height'];
  final int srcWidth = inputImageDataJson['width'];
  final int srcChannels = inputImageDataJson['channels'];
  final List<dynamic> srcDataFlatJson = inputImageDataJson['data'];
  final Uint8List srcImageHWC = Uint8List.fromList(srcDataFlatJson.cast<int>());

  final List<dynamic> mJson = testData['transformation_matrix_M'];
  final List<List<double>> matrixM = mJson.map((row) {
    return (row as List<dynamic>).map((val) => (val as num).toDouble()).toList();
  }).toList();

  final List<dynamic> dsizeJson = testData['dsize'];
  final List<int> dsize = dsizeJson.cast<int>(); // [width, height]

  final Map<String, dynamic> opencvResultJson = testData['opencv_result_image'];
  final int opencvHeight = opencvResultJson['height'];
  final int opencvWidth = opencvResultJson['width'];
  final int opencvChannels = opencvResultJson['channels'];
  final List<dynamic> opencvResultDataFlat = opencvResultJson['data'];


  print("\nThông số đầu vào từ JSON:");
  print("  Ảnh nguồn: ${srcHeight}x${srcWidth}x$srcChannels");
  print("  Ma trận M: $matrixM");
  print("  Kích thước đích (dsize): $dsize (width, height)");

  print("\nChạy hàm warpPerspectiveExactOpenCV (Dart)...");
  Stopwatch stopwatch = Stopwatch()..start();

  print("matrix M = ${matrixM.runtimeType}");

  Map<String, dynamic> resultDart = await warpPerspectiveExact(
    srcImageHWC: srcImageHWC,
    srcHeight: srcHeight,
    srcWidth: srcWidth,
    srcNumChannels: srcChannels,
    M_list: matrixM,
    dsize: dsize,
    flags: CV_INTER_LANCZOS4,
    numIsolates: 0,
  );
  stopwatch.stop();
  print("Hàm Dart thực thi trong: ${stopwatch.elapsedMilliseconds} ms");

  final Uint8List dartResultImage = resultDart['data'] as Uint8List;
  final int dartResultWidth = resultDart['width'] as int;
  final int dartResultHeight = resultDart['height'] as int;
  final int dartResultChannels = resultDart['channels'] as int;

  print("\nSo sánh kết quả Dart với kết quả OpenCV (từ JSON):");
  compareResultsDart(
      dartResultImage,
      dartResultHeight,
      dartResultWidth,
      dartResultChannels,
      opencvResultDataFlat,
      "Warp Perspective Dart vs OpenCV"
  );
}