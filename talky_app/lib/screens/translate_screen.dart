import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../app_state.dart';
import '../app_theme.dart';
import '../hand_painter.dart';
import '../hand_source.dart';
import '../hand_source_factory.dart';
import '../sibi_classifier.dart';
import '../sibi_normalizer.dart';

class TranslateScreen extends StatefulWidget {
  const TranslateScreen({super.key});

  @override
  State<TranslateScreen> createState() => _TranslateScreenState();
}

class _TranslateScreenState extends State<TranslateScreen> {
  // Samakan dengan data latih: true jika 01_collect_data.py memakai cv2.flip.
  static const bool cerminUntukModel = true;
  static const double ambangYakin = 0.70;
  static const int jendela = 4;
  static const int minSuara = 3;
  static const int tahanMs = 500;

  final SibiClassifier _classifier = SibiClassifier();
  final HandSource _source = createHandSource();
  Timer? _timer;

  bool _aktif = false;
  bool _memulai = false;
  bool _depan = true;
  String? _pesanError;

  String _hasil = '';
  String _hurufTerakhir = '-';

  List<Offset>? _titik;
  String? _hurufLive;
  double _yakinLive = 0;
  double _progres = 0;

  final List<String> _suara = [];
  String? _kandidat;
  DateTime? _kandidatSejak;
  bool _sudahDitulis = false;

  @override
  void dispose() {
    _timer?.cancel();
    _source.stop();
    tanganTerdeteksi.value = false;
    super.dispose();
  }

  void _info(String pesan) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(pesan)));
  }

  Future<void> _mulai() async {
    setState(() {
      _memulai = true;
      _pesanError = null;
    });
    try {
      if (!_classifier.isReady) await _classifier.init();
      await _source.start(depan: _depan);
      _timer?.cancel();
      _timer = Timer.periodic(const Duration(milliseconds: 70), (_) => _proses());
      if (mounted) setState(() => _aktif = true);
    } catch (e) {
      _pesanError = 'Gagal membuka kamera: $e\n'
          'Pastikan izin kamera diberikan (ikon gembok di address bar Chrome).';
    } finally {
      if (mounted) setState(() => _memulai = false);
    }
  }

  Future<void> _berhenti() async {
    _timer?.cancel();
    await _source.stop();
    _resetPengenalan();
    tanganTerdeteksi.value = false;
    if (mounted) {
      setState(() {
        _aktif = false;
        _titik = null;
        _hurufLive = null;
      });
    }
  }

  Future<void> _gantiKamera() async {
    if (!_aktif) {
      _info('Nyalakan kamera dulu');
      return;
    }
    _depan = !_depan;
    await _berhenti();
    await _mulai();
  }

  void _resetPengenalan() {
    _suara.clear();
    _kandidat = null;
    _kandidatSejak = null;
    _sudahDitulis = false;
    _progres = 0;
  }

  void _proses() {
    if (!_aktif && !_memulai) return;
    final mentah = _source.detect();
    if (mentah == null || mentah.length != 63) {
      _tanganHilang();
      return;
    }

    final titik = <Offset>[];
    final pts = <Landmark3D>[];
    for (var i = 0; i < 21; i++) {
      final x = mentah[i * 3], y = mentah[i * 3 + 1], z = mentah[i * 3 + 2];
      titik.add(Offset(x, y));
      pts.add(Landmark3D(x: cerminUntukModel ? 1 - x : x, y: y, z: z));
    }

    final norm = SibiNormalizer.normalize(pts);
    final r = norm == null ? null : _classifier.predict(norm);
    if (r == null) {
      _tanganHilang();
      return;
    }

    tanganTerdeteksi.value = true;
    _stabilkan(r);
    if (!mounted) return;
    setState(() {
      _titik = titik;
      _hurufLive = r.letter;
      _yakinLive = r.confidence;
    });
  }

  void _tanganHilang() {
    tanganTerdeteksi.value = false;
    if (_titik == null && _kandidat == null) return;
    _resetPengenalan();
    if (mounted) {
      setState(() {
        _titik = null;
        _hurufLive = null;
      });
    }
  }

  void _stabilkan(PredictionResult r) {
    _suara.add(r.confidence >= ambangYakin ? r.letter : '-');
    if (_suara.length > jendela) _suara.removeAt(0);

    final hitung = <String, int>{};
    for (final s in _suara) {
      hitung[s] = (hitung[s] ?? 0) + 1;
    }
    final terbanyak = hitung.entries.reduce((a, b) => a.value >= b.value ? a : b);
    if (terbanyak.key == '-' || terbanyak.value < minSuara) {
      _progres = 0;
      return;
    }

    final stabil = terbanyak.key;
    final sekarang = DateTime.now();
    if (stabil != _kandidat) {
      _kandidat = stabil;
      _kandidatSejak = sekarang;
      _sudahDitulis = false;
      _progres = 0;
      return;
    }
    final lama = sekarang.difference(_kandidatSejak!).inMilliseconds;
    _progres = _sudahDitulis ? 1 : (lama / tahanMs).clamp(0.0, 1.0);
    if (!_sudahDitulis && lama >= tahanMs) {
      _sudahDitulis = true;
      _hasil += stabil;
      _hurufTerakhir = stabil;
    }
  }

  Future<void> _salin() async {
    await Clipboard.setData(ClipboardData(text: _hasil));
    if (mounted) _info('Teks disalin');
  }

  void _simpan() {
    simpanRiwayat(_hasil);
    _info('Disimpan ke Riwayat');
  }

  void _hapus() => setState(() {
        _hasil = '';
        _hurufTerakhir = '-';
      });

  void _spasi() {
    if (_hasil.isNotEmpty && !_hasil.endsWith(' ')) {
      setState(() => _hasil += ' ');
    }
  }

  void _hapusSatu() {
    if (_hasil.isEmpty) return;
    setState(() => _hasil = _hasil.substring(0, _hasil.length - 1));
  }

  @override
  Widget build(BuildContext context) {
    final kosong = _hasil.isEmpty;
    return Column(
      children: [
        Expanded(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
            child: _kartuKamera(),
          ),
        ),
        _panelHasil(kosong),
      ],
    );
  }

  Widget _kartuKamera() {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.tealGelap,
        borderRadius: BorderRadius.circular(28),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        children: [
          if (_aktif) Positioned.fill(child: _source.buildPreview()),
          if (!_aktif) Positioned.fill(child: CustomPaint(painter: HandPainter())),
          
          if (_aktif && _hurufLive != null) _lencanaLive(),
          Positioned(
            top: 12,
            right: 12,
            child: Row(
              children: [
                _bulat(_aktif ? Icons.videocam_off_outlined : Icons.videocam_outlined,
                    _aktif ? _berhenti : _mulai),
                const SizedBox(width: 8),
                _bulat(Icons.cached, _gantiKamera),
              ],
            ),
          ),
          if (!_aktif) _ajakMulai(),
          Positioned(
            left: 0,
            right: 0,
            bottom: 14,
            child: Text(
              _aktif
                  ? 'Tahan pose sampai huruf tertulis'
                  : 'Posisikan seluruh tangan di dalam bingkai',
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white, fontSize: 12.5),
            ),
          ),
        ],
      ),
    );
  }

  Widget _bulat(IconData ikon, VoidCallback onTap) {
    return Material(
      color: Colors.black38,
      shape: const CircleBorder(),
      child: IconButton(
        icon: Icon(ikon, color: Colors.white),
        onPressed: _memulai ? null : onTap,
      ),
    );
  }

  Widget _ajakMulai() {
    return Positioned.fill(
      child: Align(
        alignment: const Alignment(0, 0.85),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_pesanError != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 0, 20, 10),
                child: Text(_pesanError!,
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.orangeAccent, fontSize: 12)),
              ),
            FilledButton.icon(
              onPressed: _memulai ? null : _mulai,
              icon: _memulai
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.videocam),
              label: Text(_memulai ? 'Memuat...' : 'Mulai kamera'),
              style: FilledButton.styleFrom(backgroundColor: AppColors.teal),
            ),
            const SizedBox(height: 28),
          ],
        ),
      ),
    );
  }

  Widget _lencanaLive() {
    return Positioned(
      top: 14,
      left: 14,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Terbaca: $_hurufLive  ${(_yakinLive * 100).toStringAsFixed(0)}%',
              style: const TextStyle(
                  color: Colors.white, fontSize: 12.5, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 4),
            SizedBox(
              width: 90,
              child: LinearProgressIndicator(
                value: _progres,
                minHeight: 4,
                color: AppColors.emas,
                backgroundColor: Colors.white24,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _panelHasil(bool kosong) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
      decoration: const BoxDecoration(
        color: AppColors.panel,
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.black26,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'HASIL TERJEMAHAN',
                style: TextStyle(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1,
                  color: Color(0xFF4F6B73),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Text(
                  'Huruf terakhir: $_hurufTerakhir',
                  style: const TextStyle(fontSize: 12.5, fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxHeight: 130),
                  child: SingleChildScrollView(
                    child: Text(
                      kosong ? '...' : _hasil,
                      style: TextStyle(
                        fontSize: 56,
                        fontWeight: FontWeight.w800,
                        color: kosong ? Colors.black26 : AppColors.teks,
                      ),
                    ),
                  ),
                ),
              ),
              Column(
                children: [
                  IconButton(
                    tooltip: 'Spasi',
                    icon: const Icon(Icons.space_bar),
                    onPressed: kosong ? null : _spasi,
                  ),
                  IconButton(
                    tooltip: 'Hapus satu huruf',
                    icon: const Icon(Icons.backspace_outlined),
                    onPressed: kosong ? null : _hapusSatu,
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(child: _tombol(Icons.copy, 'Salin', kosong ? null : _salin)),
              const SizedBox(width: 10),
              Expanded(
                child: _tombol(
                    Icons.bookmark_border, 'Simpan', kosong ? null : _simpan,
                    utama: true),
              ),
              const SizedBox(width: 10),
              Expanded(
                  child: _tombol(Icons.delete_outline, 'Hapus', kosong ? null : _hapus)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _tombol(IconData ikon, String label, VoidCallback? onTap,
      {bool utama = false}) {
    final bentuk = RoundedRectangleBorder(borderRadius: BorderRadius.circular(16));
    const padding = EdgeInsets.symmetric(horizontal: 6);
    if (utama) {
      return FilledButton.icon(
        onPressed: onTap,
        icon: Icon(ikon, size: 20),
        label: Text(label),
        style: FilledButton.styleFrom(
          backgroundColor: AppColors.teal,
          minimumSize: const Size(0, 52),
          padding: padding,
          shape: bentuk,
          textStyle: const TextStyle(fontWeight: FontWeight.w700),
        ),
      );
    }
    return OutlinedButton.icon(
      onPressed: onTap,
      icon: Icon(ikon, size: 20),
      label: Text(label),
      style: OutlinedButton.styleFrom(
        backgroundColor: Colors.white,
        foregroundColor: AppColors.teks,
        side: const BorderSide(color: Color(0xFFD5DEE2)),
        minimumSize: const Size(0, 52),
        padding: padding,
        shape: bentuk,
        textStyle: const TextStyle(fontWeight: FontWeight.w700),
      ),
    );
  }
}