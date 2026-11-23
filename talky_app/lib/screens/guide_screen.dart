import 'package:flutter/material.dart';
import '../app_theme.dart';

class GuideScreen extends StatefulWidget {
  const GuideScreen({super.key});

  @override
  State<GuideScreen> createState() => _GuideScreenState();
}

class _GuideScreenState extends State<GuideScreen> {
  static const List<String> _huruf = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
  ];
  static const Set<String> _gerak = {'J', 'Z'};

  // Deskripsi huruf lain diisi setelah kamu punya acuan SIBI yang benar.
  static const Map<String, String> _deskripsi = {
    'J': 'Ujung jari kelingking membentuk pose menggantung, lalu bergerak '
        'melengkung seperti kail ikan (lintasan huruf J).',
    'Z': 'Ujung jari telunjuk menunjuk ke depan, lalu bergerak zig-zag '
        '(horizontal ke kanan, diagonal ke kiri bawah, horizontal ke kanan).',
  };

  int _filter = 0; // 0 = semua, 1 = statis, 2 = gerak
  String _cari = '';

  List<String> get _tampil => _huruf.where((h) {
        final gerak = _gerak.contains(h);
        if (_filter == 1 && gerak) return false;
        if (_filter == 2 && !gerak) return false;
        return _cari.isEmpty || h.contains(_cari);
      }).toList();

  // Memakai assets/panduan/<HURUF>.png bila ada, kalau tidak ikon tangan.
  Widget _gambar(String huruf, double ukuran) {
    return Image.asset(
      'assets/panduan/$huruf.jpg',
      fit: BoxFit.contain,
      errorBuilder: (_, __, ___) =>
          Icon(Icons.back_hand_outlined, color: AppColors.teal, size: ukuran),
    );
  }

  void _buka(String huruf) {
    final gerak = _gerak.contains(huruf);
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.panel,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
      builder: (_) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 160,
                height: 160,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Center(child: _gambar(huruf, 72)),
              ),
              const SizedBox(height: 16),
              Text('Huruf $huruf',
                  style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w800,
                      color: AppColors.teks)),
              const SizedBox(height: 6),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(gerak ? 'Gerak' : 'Statis',
                    style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: AppColors.teal)),
              ),
              const SizedBox(height: 12),
              Text(
                _deskripsi[huruf] ?? 'Deskripsi pose menyusul.',
                textAlign: TextAlign.center,
              ),
              if (gerak) ...[
                const SizedBox(height: 8),
                const Text(
                  'Huruf ini berupa gerakan dan belum dikenali aplikasi pada versi ini.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.black54, fontSize: 12.5),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _kartu(String h) {
    final gerak = _gerak.contains(h);
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: () => _buka(h),
      child: Container(
        padding: const EdgeInsets.fromLTRB(10, 10, 10, 8),
        decoration: BoxDecoration(
          color: AppColors.panel,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: const Color(0xFFD5E2E6)),
        ),
        child: Column(
          children: [
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Center(child: _gambar(h, 34)),
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(h,
                    style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        color: AppColors.teks)),
                if (gerak) ...[
                  const SizedBox(width: 6),
                  const Text('Gerak',
                      style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          color: AppColors.teal)),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _chip(int nilai, String label) {
    final pilih = _filter == nilai;
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: ChoiceChip(
        label: Text(label),
        selected: pilih,
        showCheckmark: false,
        onSelected: (_) => setState(() => _filter = nilai),
        selectedColor: AppColors.teal,
        backgroundColor: AppColors.panel,
        side: BorderSide.none,
        shape: const StadiumBorder(),
        labelStyle: TextStyle(
          fontWeight: FontWeight.w700,
          fontSize: 13,
          color: pilih ? Colors.white : AppColors.teks,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final daftar = _tampil;
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
          child: TextField(
            onChanged: (v) => setState(() => _cari = v.trim().toUpperCase()),
            decoration: InputDecoration(
              hintText: 'Cari huruf',
              prefixIcon: const Icon(Icons.search),
              filled: true,
              fillColor: AppColors.panel,
              contentPadding: EdgeInsets.zero,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
                borderSide: BorderSide.none,
              ),
            ),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          height: 38,
          child: ListView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            children: [
              _chip(0, 'Semua'),
              _chip(1, 'Statis'),
              _chip(2, 'Gerak (J, Z)'),
            ],
          ),
        ),
        const SizedBox(height: 8),
        Expanded(
          child: daftar.isEmpty
              ? const Center(
                  child: Text('Huruf tidak ditemukan',
                      style: TextStyle(color: Colors.black54)))
              : GridView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: daftar.length,
                  gridDelegate:
                      const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 3,
                    crossAxisSpacing: 12,
                    mainAxisSpacing: 12,
                    childAspectRatio: 0.85,
                  ),
                  itemBuilder: (_, i) => _kartu(daftar[i]),
                ),
        ),
      ],
    );
  }
}