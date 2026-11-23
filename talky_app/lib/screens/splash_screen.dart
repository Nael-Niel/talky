import 'dart:async';

import 'package:flutter/material.dart';

import '../app_theme.dart';

class SplashScreen extends StatefulWidget {
  /// Halaman yang dibuka setelah splash selesai.
  final Widget tujuan;

  const SplashScreen({super.key, required this.tujuan});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  static const int _lamaMs = 2200; // lama splash sebelum pindah halaman
  static const String _versi = 'Versi 1.0'; // ubah manual bila versi berganti

  late final AnimationController _kontrol = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 800),
  );
  late final Animation<double> _muncul =
      CurvedAnimation(parent: _kontrol, curve: Curves.easeOut);
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _kontrol.forward();
    _timer = Timer(const Duration(milliseconds: _lamaMs), _lanjut);
  }

  @override
  void dispose() {
    _timer?.cancel();
    _kontrol.dispose();
    super.dispose();
  }

  void _lanjut() {
    if (!mounted) return;
    _timer?.cancel();
    Navigator.of(context).pushReplacement(
      PageRouteBuilder(
        pageBuilder: (_, __, ___) => widget.tujuan,
        transitionsBuilder: (_, anim, __, child) =>
            FadeTransition(opacity: anim, child: child),
        transitionDuration: const Duration(milliseconds: 400),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.teal,
      body: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: _lanjut, // ketuk layar untuk melewati splash
        child: SafeArea(
          child: FadeTransition(
            opacity: _muncul,
            child: Column(
              children: [
                Expanded(
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 96,
                          height: 96,
                          decoration: const BoxDecoration(
                            color: Colors.white,
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(
                            Icons.back_hand_outlined,
                            color: AppColors.teal,
                            size: 52,
                          ),
                        ),
                        const SizedBox(height: 22),
                        const Text(
                          'Talky',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 38,
                            fontWeight: FontWeight.w800,
                            letterSpacing: -0.5,
                          ),
                        ),
                        const SizedBox(height: 14),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 48),
                          child: Text(
                            'Terjemahkan abjad isyarat SIBI\nmenjadi teks',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: Colors.white.withValues(alpha: 0.92),
                              fontSize: 15,
                              height: 1.4,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(bottom: 28),
                  child: Text(
                    _versi,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.85),
                      fontSize: 12.5,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}