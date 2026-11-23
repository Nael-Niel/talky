import 'package:flutter/foundation.dart';

class RiwayatItem {
  final String teks;
  final DateTime waktu;
  RiwayatItem(this.teks, this.waktu);
}

final ValueNotifier<List<RiwayatItem>> riwayat = ValueNotifier([]);

final ValueNotifier<bool> tanganTerdeteksi = ValueNotifier(false);

void simpanRiwayat(String teks) {
  riwayat.value = [RiwayatItem(teks, DateTime.now()), ...riwayat.value];
}

void hapusRiwayat(int index) {
  final daftar = [...riwayat.value]..removeAt(index);
  riwayat.value = daftar;
}