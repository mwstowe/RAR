# RAR Rust
This crate provides a Rust native functionality to list and extract RAR files (Right now with limited functionality!)

Please have a look in the test section of the file `src/lib.rs` to see in detail which features are supported right now and how to use this crate.

A basic example to extract the complete archive:
```rust
extern crate rar;

// Get the archive information and extract everything
let archive = rar::Archive::extract_all(
    "assets/rar5-save-32mb-txt.rar",
    "target/rar-test/rar5-save-32mb-txt/",
    "").unwrap();

// Print out the archive structure information
println!("Result: {:?}", archive);
```

## Version 0.4.1
This version includes:
- **Fixed compression flag detection** for RAR5 archives
- Updated dependencies for better compatibility
- Fixed CBC decryption implementation for encrypted archives
- Improved code quality and Rust idioms
- Basic compression framework added (partial implementation)
- All tests now pass (35/35) for uncompressed files

# Features
**RAR 5**
- [x] Extract archive with single File
- [x] Extract archive with multiple Files
- [x] Extract split archive with multiple files
- [x] Extract encrypted archive
- [x] Extract compression SAVE
- [ ] Extract compression FASTEST (partial - framework in place)
- [ ] Extract compression FAST (partial - framework in place)
- [ ] Extract compression NORMAL (partial - framework in place)
- [ ] Extract compression GOOD (partial - framework in place)
- [ ] Extract compression BEST (partial - framework in place)

**RAR 4**
- [ ] Extract archive with single File
- [ ] Extract archive with multiple Files
- [ ] Extract split archive with multiple files
- [ ] Extract encrypted archive
- [ ] Extract compression SAVE
- [ ] Extract compression FASTEST
- [ ] Extract compression FAST
- [ ] Extract compression NORMAL
- [ ] Extract compression GOOD
- [ ] Extract compression BEST

# Contributing
Please contribute! 

The goal is to make this crate feature complete :)

If you need any kind of help, open an issue or write me an mail.
Pull requests are welcome!

## Compression Implementation Status
The crate now includes a compression framework in `src/compression.rs` with **fixed compression detection**. The RAR compression algorithms are proprietary and complex, requiring significant reverse engineering effort to implement fully. The current implementation provides:

- Framework for handling different compression types
- Support for uncompressed (SAVE) files
- **Fixed compression flag detection** for RAR5 archives
- Error handling for compressed files (returns UnsupportedCompression error)

### Recent Fixes
- **Fixed compression flag parsing**: The compression method is now correctly extracted from bits 7-10 of the compression vint value
- **Verified with RAR 7.12**: Tested with archives created using different compression levels (-m0 through -m5)
- **Proper error handling**: Compressed files now correctly return `UnsupportedCompression` error instead of being misidentified as uncompressed

To fully implement RAR compression support, contributors would need to:
1. ✅ ~~Fix compression flag detection for RAR5 archives~~ **COMPLETED**
2. Reverse engineer the RAR compression algorithms
3. Implement the LZ77-based compression with RAR-specific optimizations
4. Handle the various compression levels (FASTEST, FAST, NORMAL, GOOD, BEST)

# License
Copyright © 2018 Robert Schütte

Distributed under the [MIT License](LICENSE).
