import 'package:flutter/widgets.dart';
import 'hand_source.dart';

class _StubHandSource implements HandSource {
  @override
  Future<void> start({required bool depan}) async {
    throw UnsupportedError(
        'Kamera di platform ini belum tersedia. Gunakan Chrome untuk sementara.');
  }

  @override
  Future<void> stop() async {}

  @override
  Size get videoSize => Size.zero;

  @override
  Widget buildPreview() => const SizedBox.shrink();

  @override
  List<double>? detect() => null;
}

HandSource createHandSource() => _StubHandSource();