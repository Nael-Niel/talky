import 'dart:math';
import 'package:flutter/material.dart';
import 'app_theme.dart';

const List<List<int>> _garisTangan = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

void _gambar(Canvas canvas, Offset Function(int) p) {
  final garis = Paint()
    ..color = AppColors.emas
    ..strokeWidth = 3
    ..strokeCap = StrokeCap.round;
  for (final g in _garisTangan) {
    canvas.drawLine(p(g[0]), p(g[1]), garis);
  }
  final isi = Paint()..color = Colors.white;
  final tepi = Paint()
    ..color = AppColors.emas
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2.5;
  for (var i = 0; i < 21; i++) {
    canvas.drawCircle(p(i), 5.5, isi);
    canvas.drawCircle(p(i), 5.5, tepi);
  }
}

class HandPainter extends CustomPainter {
  static const List<Offset> _titik = [
    Offset(185, 379),
    Offset(153, 347), Offset(133, 311), Offset(121, 278), Offset(113, 250),
    Offset(173, 311), Offset(170, 268), Offset(168, 227), Offset(167, 198),
    Offset(193, 307), Offset(196, 270), Offset(199, 228), Offset(201, 182),
    Offset(213, 314), Offset(222, 264), Offset(229, 230), Offset(234, 206),
    Offset(232, 330), Offset(250, 290), Offset(262, 266), Offset(272, 245),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    var minX = _titik.first.dx, maxX = minX;
    var minY = _titik.first.dy, maxY = minY;
    for (final t in _titik) {
      minX = min(minX, t.dx);
      maxX = max(maxX, t.dx);
      minY = min(minY, t.dy);
      maxY = max(maxY, t.dy);
    }
    final w = maxX - minX, h = maxY - minY;
    final skala = min(size.width * 0.62 / w, size.height * 0.66 / h);
    final offX = (size.width - w * skala) / 2 - minX * skala;
    final offY =
        (size.height - h * skala) / 2 - minY * skala - size.height * 0.04;
    _gambar(canvas,
        (i) => Offset(_titik[i].dx * skala + offX, _titik[i].dy * skala + offY));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class LiveHandPainter extends CustomPainter {
  final List<Offset> titik;
  final Size videoSize;
  final bool cermin;

  LiveHandPainter(this.titik, this.videoSize, this.cermin);

  @override
  void paint(Canvas canvas, Size size) {
    if (titik.length != 21) return;
    final vw = videoSize.width > 0 ? videoSize.width : size.width;
    final vh = videoSize.height > 0 ? videoSize.height : size.height;
    final skala = max(size.width / vw, size.height / vh);
    final dw = vw * skala, dh = vh * skala;
    final ox = (size.width - dw) / 2, oy = (size.height - dh) / 2;
    Offset p(int i) {
      final x = cermin ? 1 - titik[i].dx : titik[i].dx;
      return Offset(ox + x * dw, oy + titik[i].dy * dh);
    }

    _gambar(canvas, p);
  }

  @override
  bool shouldRepaint(covariant LiveHandPainter old) => true;
}