import 'package:flutter/material.dart';
import 'app_state.dart';
import 'app_theme.dart';
import 'screens/splash_screen.dart';
import 'screens/guide_screen.dart';
import 'screens/history_screen.dart';
import 'screens/translate_screen.dart';

void main() {
  runApp(const TalkyApp());
}

class TalkyApp extends StatelessWidget {
  const TalkyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Talky',
      theme: appTheme(),
       home: const SplashScreen(tujuan: AppShell()),
    );
  }
}

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _index = 0;
  static const _judul = ['Terjemah', 'Panduan SIBI', 'Riwayat'];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_judul[_index]),
        actions: [
          if (_index == 0) const _StatusTangan(),
          const SizedBox(width: 16),
        ],
      ),
      drawer: Drawer(
        child: SafeArea(
          child: ListView(
            children: [
              const ListTile(
                title: Text('Talky',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800)),
                subtitle: Text('Penerjemah abjad SIBI'),
              ),
              const Divider(),
              ListTile(
                leading: const Icon(Icons.info_outline),
                title: const Text('Tentang'),
                onTap: () {
                  Navigator.pop(context);
                  showAboutDialog(
                    context: context,
                    applicationName: 'Talky',
                    children: const [
                      Text('Menerjemahkan abjad SIBI menjadi teks. '
                          'Proyek Tugas Akhir D3 Teknik Informatika.'),
                    ],
                  );
                },
              ),
            ],
          ),
        ),
      ),
      body: IndexedStack(
        index: _index,
        children: const [TranslateScreen(), GuideScreen(), HistoryScreen()],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: const [
          NavigationDestination(
              icon: Icon(Icons.photo_camera_outlined),
              selectedIcon: Icon(Icons.photo_camera),
              label: 'Terjemah'),
          NavigationDestination(
              icon: Icon(Icons.article_outlined),
              selectedIcon: Icon(Icons.article),
              label: 'Panduan'),
          NavigationDestination(icon: Icon(Icons.history), label: 'Riwayat'),
        ],
      ),
    );
  }
}

class _StatusTangan extends StatelessWidget {
  const _StatusTangan();

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: tanganTerdeteksi,
      builder: (context, ada, _) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
          decoration: BoxDecoration(
            color: AppColors.chip,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  color: ada ? AppColors.hijau : Colors.grey,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                ada ? 'Tangan terdeteksi' : 'Tangan tidak terdeteksi',
                style: const TextStyle(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: AppColors.teks),
              ),
            ],
          ),
        );
      },
    );
  }
}