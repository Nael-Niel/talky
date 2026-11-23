import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../app_state.dart';
import '../app_theme.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  String _waktu(DateTime t) {
    String d(int n) => n.toString().padLeft(2, '0');
    return '${d(t.day)}/${d(t.month)}/${t.year}  ${d(t.hour)}:${d(t.minute)}';
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<List<RiwayatItem>>(
      valueListenable: riwayat,
      builder: (context, daftar, _) {
        if (daftar.isEmpty) {
          return const Center(
            child: Text('Belum ada riwayat.\nTekan Simpan di halaman Terjemah.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.black54)),
          );
        }
        return ListView.separated(
          padding: const EdgeInsets.all(16),
          itemCount: daftar.length,
          separatorBuilder: (_, __) => const SizedBox(height: 10),
          itemBuilder: (context, i) {
            final item = daftar[i];
            return Container(
              decoration: BoxDecoration(
                color: AppColors.panel,
                borderRadius: BorderRadius.circular(16),
              ),
              child: ListTile(
                title: Text(item.teks,
                    style: const TextStyle(
                        fontSize: 20, fontWeight: FontWeight.w800)),
                subtitle: Text(_waktu(item.waktu)),
                trailing: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      icon: const Icon(Icons.copy),
                      onPressed: () =>
                          Clipboard.setData(ClipboardData(text: item.teks)),
                    ),
                    IconButton(
                      icon: const Icon(Icons.delete_outline),
                      onPressed: () => hapusRiwayat(i),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }
}