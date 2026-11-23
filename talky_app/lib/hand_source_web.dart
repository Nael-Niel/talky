import 'dart:convert';
import 'dart:js_interop';
import 'dart:ui_web' as ui_web;

import 'package:flutter/widgets.dart';

import 'hand_source.dart';

@JS('talkyHand')
external _TalkyHand? get _talkyHand;

extension type _TalkyHand._(JSObject _) implements JSObject {
  external JSPromise<JSAny?> init(JSString modelUrl);
  external JSPromise<JSAny?> start(JSString facing);
  external void stop();
  external JSString detect();
  external JSString size();
  external JSObject get video;
}

const _viewType = 'talky-video';
bool _terdaftar = false;

class _WebHandSource implements HandSource {
  Size _ukuran = Size.zero;

  @override
  Future<void> start({required bool depan}) async {
    final t = _talkyHand;
    if (t == null) {
      throw StateError(
          'Pustaka MediaPipe belum termuat. Periksa koneksi internet lalu muat ulang halaman.');
    }
    if (!_terdaftar) {
      ui_web.platformViewRegistry
          .registerViewFactory(_viewType, (int id) => _talkyHand!.video);
      _terdaftar = true;
    }
    await t.init('assets/assets/model/hand_landmarker.task'.toJS).toDart;
    await t.start((depan ? 'user' : 'environment').toJS).toDart;
  }

  @override
  Future<void> stop() async {
    _talkyHand?.stop();
    _ukuran = Size.zero;
  }

  @override
  Size get videoSize {
    if (_ukuran == Size.zero) {
      final t = _talkyHand;
      if (t != null) {
        final bagian = t.size().toDart.split('x');
        if (bagian.length == 2) {
          final w = double.tryParse(bagian[0]) ?? 0;
          final h = double.tryParse(bagian[1]) ?? 0;
          if (w > 0 && h > 0) _ukuran = Size(w, h);
        }
      }
    }
    return _ukuran;
  }

  @override
  Widget buildPreview() => const HtmlElementView(viewType: _viewType);

  @override
  List<double>? detect() {
    final t = _talkyHand;
    if (t == null) return null;
    final s = t.detect().toDart;
    if (s.isEmpty) return null;
    return (jsonDecode(s) as List).map((e) => (e as num).toDouble()).toList();
  }
}

HandSource createHandSource() => _WebHandSource();