import 'package:flutter/widgets.dart';

abstract class HandSource {
  Future<void> start({required bool depan});
  Future<void> stop();
  Size get videoSize;
  Widget buildPreview();
  List<double>? detect();
}