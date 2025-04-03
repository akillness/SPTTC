export 'unsupported.dart' // Stub implementation
    if (dart.library.ffi) 'native.dart' // VM implementation
    if (dart.library.html) 'web.dart'; // Web implementation 