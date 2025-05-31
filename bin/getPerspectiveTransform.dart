import 'dart:math';
import 'package:equations/equations.dart' as equations;
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

// ==================== PRIVATE HELPER CLASSES ====================

class Mt {
  /// Tạo ma trận toàn số 0
  static Matrix zeros(int rows, int cols, {DType dtype = DType.float64}) {
    if (rows <= 0 || cols <= 0) {
      throw ArgumentError('Số hàng và số cột phải lớn hơn 0.');
    }
    final flatData = List.filled(rows * cols, 0.0);
    return Matrix.fromFlattenedList(flatData, rows, cols, dtype: dtype);
  }

  /// Tạo ma trận với các số nguyên liên tiếp
  static Matrix intNumbers(int rows, int cols,
      {bool startAtOne = false, DType dtype = DType.float64}) {
    if (rows <= 0 || cols <= 0) {
      throw ArgumentError('Số hàng và số cột phải lớn hơn 0.');
    }

    final flatData = <double>[];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        double value = (i * cols + j).toDouble();
        if (startAtOne) {
          value += 1.0;
        }
        flatData.add(value);
      }
    }

    return Matrix.fromFlattenedList(flatData, rows, cols, dtype: dtype);
  }

  /// Tạo ma trận với các giá trị ngẫu nhiên trong khoảng [min, max)
  static Matrix randomRanged(int rows, int cols, double min, double max,
      {DType dtype = DType.float64, int? seed}) {
    if (rows <= 0 || cols <= 0) {
      throw ArgumentError('Số hàng và số cột phải lớn hơn 0.');
    }
    if (min >= max) {
      throw ArgumentError('Giá trị min phải nhỏ hơn max.');
    }

    final random = Random(seed);
    final flatData = <double>[];

    for (int i = 0; i < rows * cols; i++) {
      flatData.add(min + random.nextDouble() * (max - min));
    }

    return Matrix.fromFlattenedList(flatData, rows, cols, dtype: dtype);
  }

  /// Tạo ma trận với các giá trị ngẫu nhiên trong khoảng [0.0, 1.0)
  static Matrix random(int rows, int cols,
      {DType dtype = DType.float64, int? seed}) {
    if (rows <= 0 || cols <= 0) {
      throw ArgumentError('Số hàng và số cột phải lớn hơn 0.');
    }
    return randomRanged(rows, cols, 0.0, 1.0, dtype: dtype, seed: seed);
  }

  /// Tạo ma trận với các giá trị ngẫu nhiên trong khoảng tùy chỉnh
  /// (Phương thức này giống randomRanged, có thể xóa nếu không cần thiết)
  static Matrix randomCustomRange(int rows, int cols, double overallMin,
      double overallMax, {DType dtype = DType.float64, int? seed}) {
    return randomRanged(
        rows, cols, overallMin, overallMax, dtype: dtype, seed: seed);
  }

  /// Tạo ma trận đơn vị (identity matrix)
  static Matrix eye(int size, {DType dtype = DType.float64}) {
    if (size <= 0) {
      throw ArgumentError('Kích thước ma trận phải lớn hơn 0.');
    }

    final flatData = <double>[];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        flatData.add(i == j ? 1.0 : 0.0);
      }
    }

    return Matrix.fromFlattenedList(flatData, size, size, dtype: dtype);
  }

  /// Tạo ma trận với tất cả phần tử có cùng giá trị
  static Matrix full(int rows, int cols, double value,
      {DType dtype = DType.float64}) {
    if (rows <= 0 || cols <= 0) {
      throw ArgumentError('Số hàng và số cột phải lớn hơn 0.');
    }

    final flatData = List.filled(rows * cols, value);
    return Matrix.fromFlattenedList(flatData, rows, cols, dtype: dtype);
  }

  /// Tạo ma trận từ danh sách 2D
  static Matrix fromList(List<List<double>> data,
      {DType dtype = DType.float64}) {
    if (data.isEmpty || data.first.isEmpty) {
      throw ArgumentError('Dữ liệu không được rỗng.');
    }

    final rows = data.length;
    final cols = data.first.length;

    // Kiểm tra tất cả hàng có cùng số cột
    for (int i = 0; i < rows; i++) {
      if (data[i].length != cols) {
        throw ArgumentError('Tất cả hàng phải có cùng số cột.');
      }
    }

    final flatData = <double>[];
    for (int i = 0; i < rows; i++) {
      flatData.addAll(data[i]);
    }

    return Matrix.fromFlattenedList(flatData, rows, cols, dtype: dtype);
  }

  /// Tạo ma trận từ danh sách 1D
  static Matrix fromFlatList(List<double> data, int rows, int cols,
      {DType dtype = DType.float64}) {
    if (data.length != rows * cols) {
      throw ArgumentError('Kích thước dữ liệu không khớp với số hàng và cột.');
    }

    return Matrix.fromFlattenedList(data, rows, cols, dtype: dtype);
  }

  /// Tạo ma trận với việc thiết lập giá trị tại vị trí (i,j)
  static Matrix diagonal(List<double> diagonalValues,
      {int? rows, int? cols, DType dtype = DType.float64}) {
    final size = diagonalValues.length;
    final actualRows = rows ?? size;
    final actualCols = cols ?? size;

    final flatData = <double>[];
    for (int i = 0; i < actualRows; i++) {
      for (int j = 0; j < actualCols; j++) {
        if (i == j && i < diagonalValues.length) {
          flatData.add(diagonalValues[i]);
        } else {
          flatData.add(0.0);
        }
      }
    }

    return Matrix.fromFlattenedList(
        flatData, actualRows, actualCols, dtype: dtype);
  }

  /// Tạo ma trận với khả năng thiết lập giá trị tại các vị trí cụ thể
  static Matrix withValues(int rows, int cols, Map<List<int>, double> values,
      {DType dtype = DType.float64}) {
    final flatData = List.filled(rows * cols, 0.0);

    values.forEach((position, value) {
      if (position.length == 2) {
        final i = position[0];
        final j = position[1];
        if (i >= 0 && i < rows && j >= 0 && j < cols) {
          final index = i * cols + j;
          flatData[index] = value;
        }
      }
    });

    return Matrix.fromFlattenedList(flatData, rows, cols, dtype: dtype);
  }
}

class LUDecomResult {
  Matrix P;
  Matrix L;
  Matrix U;

  LUDecomResult(this.P, this.L, this.U);
}

class warpPerspective {
  static const double _Tol = 1e-10;
  static const int _MaxRepeat = 1000;
  static const double _epsilon = 1e-10;
  // ==================== PRIVATE HELPER METHODS ====================

  Matrix _listToMatrix(List<List<double>> listData) {
    return Matrix.fromList(listData);
  }

  List<List<double>> _matrixToList(Matrix matrix) {
    return matrix.rows
        .map((vector) => vector.toList(growable: true))
        .toList(growable: true);
  }

  LUDecomResult luDecompositionManual(Matrix A_matrix) {
    final n = A_matrix.rowCount;
    var U_data = _matrixToList(A_matrix);
    var L_data = List.generate(
        n, (i) => List.generate(n, (j) => (i == j) ? 1.0 : 0.0, growable: true),
        growable: true);

    var P_data = List.generate(
        n, (i) => List.generate(n, (j) => (i == j) ? 1.0 : 0.0, growable: true),
        growable: true);

    for (int k = 0; k < n - 1; k++) {
      int pivot_row = k;
      for (int i = k + 1; i < n; i++) {
        if (U_data[i][k].abs() > U_data[pivot_row][k].abs()) {
          pivot_row = i;
        }
      }
      if (pivot_row != k) {
        var tempRowU = U_data[k];
        U_data[k] = U_data[pivot_row];
        U_data[pivot_row] = tempRowU;

        var tempRowP = P_data[k];
        P_data[k] = P_data[pivot_row];
        P_data[pivot_row] = tempRowP;

        if (k > 0) {
          for (int j = 0; j < k; j++) {
            var tempValL = L_data[k][j];
            L_data[k][j] = L_data[pivot_row][j];
            L_data[pivot_row][j] = tempValL;
          }
        }
      }
      if (U_data[k][k].abs() < 1e-12) {
        throw Exception("Matrix singular - cant deploy LU decomposition");
      }
      for (int i = k + 1; i < n; i++) {
        final multiplier = U_data[i][k] / U_data[k][k];
        L_data[i][k] = multiplier;

        for (int j = k; j < n; j++) {
          U_data[i][j] = U_data[i][j] - multiplier * U_data[k][j];
        }
      }
    }

    return LUDecomResult(
        _listToMatrix(P_data), _listToMatrix(L_data), _listToMatrix(U_data));
  }

  Vector? _solveLinearSystemLU(Matrix A, Vector B_vec, double Tol) {
    final decomposed = luDecompositionManual(A);
    final P = decomposed.P;
    final L = decomposed.L;
    final U = decomposed.U;

    Vector Pb = (P * B_vec).toVector();

    final Y_list = List<double>.filled(Pb.length, 0.0);
    // L * Y = B
    for (int i = 0; i < L.rowCount; i++) {
      double sum = 0.0;
      for (int j = 0; j < i; j++) {
        sum += L[i][j] * Y_list[j];
      }
      Y_list[i] = (Pb[i] - sum) / L[i][i];
    }

    final Vector Y_vec = Vector.fromList(Y_list, dtype: B_vec.dtype);

    final X_list = List<double>.filled(Y_vec.length, 0.0);
    for (int i = U.rowCount - 1; i >= 0; i--) {
      double sum = 0.0;
      for (int j = i + 1; j < U.columnCount; j++) {
        sum += U[i][j] * X_list[j];
      }
      X_list[i] = (Y_vec[i] - sum) / U[i][i];
    }
    final Vector X_solution_vec = Vector.fromList(X_list, dtype: B_vec.dtype);

    Vector residual = ((A * X_solution_vec).toVector() - B_vec);
    double normVal = residual.norm();

    if (normVal < Tol) {
      return X_solution_vec;
    } else {
      return null;
    }
  }
  Matrix getPerspectiveTransform(List<Point> src, List<Point> dst) {
    if (src.length != 4 || dst.length != 4) {
      throw ArgumentError(
          "Source and destination lists must both contain 4 points.");
    }
    // try c22 = 1
    var aData = List.generate(8, (_) => List<double>.filled(8, 0.0));
    var bData = List<double>.filled(8, 0.0);

    for (int i = 0; i < 4; ++i) {
      double sx = src[i].x.toDouble();
      double sy = src[i].y.toDouble();
      double dx = dst[i].x.toDouble();
      double dy = dst[i].y.toDouble();

      // Row for x'
      aData[i][0] = sx;
      aData[i][1] = sy;
      aData[i][2] = 1.0;
      aData[i][3] = 0.0;
      aData[i][4] = 0.0;
      aData[i][5] = 0.0;
      aData[i][6] = -sx * dx;
      aData[i][7] = -sy * dx;
      bData[i] = dx;

      // Row for y'
      aData[i + 4][0] = 0.0;
      aData[i + 4][1] = 0.0;
      aData[i + 4][2] = 0.0;
      aData[i + 4][3] = sx;
      aData[i + 4][4] = sy;
      aData[i + 4][5] = 1.0;
      aData[i + 4][6] = -sx * dy;
      aData[i + 4][7] = -sy * dy;
      bData[i + 4] = dy;
    }

    Matrix A = Matrix.fromList(aData, dtype: DType.float64); // A (8x8)
    Vector B_vec = Vector.fromList(bData, dtype: DType.float64); // B (8x1)
    Vector? X8_vec = _solveLinearSystemLU(A, B_vec, 1e-8);

    if (X8_vec != null) {
      return Matrix.fromList([
        [X8_vec[0], X8_vec[1], X8_vec[2]],
        [X8_vec[3], X8_vec[4], X8_vec[5]],
        [X8_vec[6], X8_vec[7], 1.0],
      ], dtype: DType.float64);
    } else {
      List<List<double>> A_sv = [];
      int numPoints = src.length;

      for (int i = 0; i < numPoints; i++) {
        double x = src[i].x.toDouble();
        double y = src[i].y.toDouble();
        double xp = dst[i].x.toDouble();
        double yp = dst[i].y.toDouble();

        A_sv.add([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]);
        A_sv.add([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]);
      }
      final matrixA = equations.RealMatrix.fromData(
          columns: 9,
          rows: 8,
          data: A_sv
      );
      final svd = matrixA.singleValueDecomposition();
      final h = svd.reversed.first.transpose().toListOfList()[8];
      return Matrix.fromList([
        [h[0] / h[8], h[1] / h[8], h[2] / h[8]],
        [h[3] / h[8], h[4] / h[8], h[5] / h[8]],
        [h[6] / h[8], h[7] / h[8], h[8] / h[8]],
      ], dtype: DType.float64);
    }
  }
}


/// ======================== TEST ============================
void main() {
  List<Point> srcPoints = [
    Point(50, 50),
    Point(200, 50),
    Point(50, 200),
    Point(200, 200),
  ];

  List<Point> dstPoints = [
    Point(10, 100),
    Point(200, 50),
    Point(100, 250),
    Point(300, 200),
  ];

  print("--- Test with regular points ---");
  try {
    Matrix perspectiveMatrix = warpPerspective().getPerspectiveTransform(
        srcPoints, dstPoints);
    print("Perspective Transform Matrix M (từ LU hoặc SVD):");
    print(perspectiveMatrix);

    Point testSrc = srcPoints[0];
    Point expectedDst = dstPoints[0];

    Vector srcVec = Vector.fromList(
        [testSrc.x, testSrc.y, 1.0], dtype: DType.float64);
    Vector dstHomogeneous = (perspectiveMatrix * srcVec).toVector();

    print("dstHomogeneous = $dstHomogeneous");

    double u_calc, v_calc;
    if (dstHomogeneous[2].abs() < 1e-9) {
      // Avoid division by zero
      u_calc = dstHomogeneous[0] / (dstHomogeneous[2].sign * 1e-9);
      v_calc = dstHomogeneous[1] / (dstHomogeneous[2].sign * 1e-9);
      print("Warning: Denominator in perspective transform is near zero.");
    } else {
      u_calc = dstHomogeneous[0] / dstHomogeneous[2];
      v_calc = dstHomogeneous[1] / dstHomogeneous[2];
    }

    print("\nTest transform for src[0] = ${testSrc}:");
    print("Expected dst[0]: ${expectedDst}");
    print(
        "Calculated (u,v): Point(x: ${u_calc.toStringAsFixed(5)}, y: ${v_calc
            .toStringAsFixed(5)})");
    print(
        "Difference: Point(x: ${(u_calc - expectedDst.x).abs().toStringAsFixed(
            5)}, y: ${(v_calc - expectedDst.y).abs().toStringAsFixed(5)})");
  } catch (e, s) {
    print("An error occurred: $e");
    print("Stack trace: $s");
  }
}
