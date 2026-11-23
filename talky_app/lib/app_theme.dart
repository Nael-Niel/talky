import 'package:flutter/material.dart';

class AppColors {
  static const teal = Color(0xFF0F6B74);
  static const tealGelap = Color(0xFF1D3A3F);
  static const panel = Color(0xFFEAF1F3);
  static const chip = Color(0xFFE6EEF1);
  static const teks = Color(0xFF12303A);
  static const emas = Color(0xFFF2B53B);
  static const hijau = Color(0xFF2EAD62);
}

ThemeData appTheme() {
  return ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: AppColors.teal,
      surface: Colors.white,
    ),
    scaffoldBackgroundColor: Colors.white,
    appBarTheme: const AppBarTheme(
      backgroundColor: Colors.white,
      surfaceTintColor: Colors.transparent,
      foregroundColor: AppColors.teks,
      titleTextStyle: TextStyle(
        fontSize: 22,
        fontWeight: FontWeight.w800,
        color: AppColors.teks,
      ),
    ),
    navigationBarTheme: const NavigationBarThemeData(
      backgroundColor: Colors.white,
      surfaceTintColor: Colors.transparent,
      indicatorColor: Color(0xFFD5E9EC),
    ),
  );
}