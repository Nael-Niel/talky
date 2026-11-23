import 'dart:math';

class Landmark3D {
  final double x;
  final double y;
  final double z;

  const Landmark3D({
    required this.x,
    required this.y,
    required this.z,
  });
}

class SibiNormalizer {
  static List<double>? normalize(List<Landmark3D> landmarks) {
    if (landmarks.length != 21) {
      return null;
    }

    final wrist = landmarks[0];
    final translated = List<Landmark3D>.generate(21, (i) {
      return Landmark3D(
        x: landmarks[i].x - wrist.x,
        y: landmarks[i].y - wrist.y,
        z: landmarks[i].z - wrist.z,
      );
    });

    double maxDistance = 0.0;
    for (final pt in translated) {
      final dist = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
      if (dist > maxDistance) {
        maxDistance = dist;
      }
    }

    if (maxDistance < 1e-6) {
      return null;
    }

    final List<double> normalized63 = [];
    for (final pt in translated) {
      normalized63.add(pt.x / maxDistance);
      normalized63.add(pt.y / maxDistance);
      normalized63.add(pt.z / maxDistance);
    }

    return normalized63;
  }
}