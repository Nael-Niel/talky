import 'dart:convert';
import 'dart:math';

import 'package:flutter/services.dart';

class PredictionResult {
  final String letter;
  final double confidence;
  final int classIndex;
  final List<MapEntry<String, double>> top3;

  PredictionResult({
    required this.letter,
    required this.confidence,
    required this.classIndex,
    required this.top3,
  });

  @override
  String toString() {
    final atas = top3
        .map((e) => '${e.key} ${(e.value * 100).toStringAsFixed(1)}%')
        .join('  |  ');
    return 'Huruf: $letter (${(confidence * 100).toStringAsFixed(1)}%)\n3 teratas: $atas';
  }
}

class _Layer {
  final List<List<double>> w; // [keluaran][masukan]
  final List<double> b;
  final bool relu;
  _Layer(this.w, this.b, this.relu);
}

/// MLP dihitung langsung di Dart (63 -> 256 -> 128 -> 24 + softmax).
/// Tanpa tflite_flutter, sehingga jalan di Chrome, Windows, dan Android.
class SibiClassifier {
  List<String> _labels = [];
  List<_Layer> _layers = [];

  bool get isReady => _layers.isNotEmpty && _labels.isNotEmpty;

  Future<void> init({
    String weightsAssetPath = 'assets/model/model_weights.json',
  }) async {
    final raw = await rootBundle.loadString(weightsAssetPath);
    final data = jsonDecode(raw) as Map<String, dynamic>;

    _labels = (data['labels'] as List).cast<String>();
    _layers = (data['layers'] as List).map((l) {
      final w = (l['w'] as List)
          .map((row) =>
              (row as List).map((v) => (v as num).toDouble()).toList())
          .toList();
      final b = (l['b'] as List).map((v) => (v as num).toDouble()).toList();
      return _Layer(w, b, l['relu'] as bool);
    }).toList();
  }

  PredictionResult? predict(List<double> normalized63) {
    if (!isReady || normalized63.length != 63) return null;

    var x = normalized63;
    for (final layer in _layers) {
      final out = List<double>.filled(layer.b.length, 0.0);
      for (var j = 0; j < out.length; j++) {
        final row = layer.w[j];
        var sum = layer.b[j];
        for (var i = 0; i < x.length; i++) {
          sum += row[i] * x[i];
        }
        out[j] = (layer.relu && sum < 0) ? 0.0 : sum;
      }
      x = out;
    }

    // softmax
    final maxV = x.reduce(max);
    final exps = x.map((v) => exp(v - maxV)).toList();
    final total = exps.reduce((a, b) => a + b);
    final probs = exps.map((e) => e / total).toList();

    final idx = List<int>.generate(probs.length, (i) => i)
      ..sort((a, b) => probs[b].compareTo(probs[a]));
    final best = idx.first;

    return PredictionResult(
      letter: _labels[best],
      confidence: probs[best],
      classIndex: best,
      top3: idx.take(3).map((i) => MapEntry(_labels[i], probs[i])).toList(),
    );
  }

  void close() {}
}
