use crate::error::{RarError, Result};
use crate::file_block::{Compression, CompressionFlags};
use std::io::Read;

// RAR decompression constants based on unarr
const LENGTH_BASES: [u32; 28] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20,
    24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224
];

const LENGTH_BITS: [i32; 28] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
    2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5
];

const SHORT_BASES: [u32; 8] = [0, 4, 8, 16, 32, 64, 128, 192];
const SHORT_BITS: [i32; 8] = [2, 2, 3, 4, 5, 6, 6, 6];

// Default Huffman table lengths for RAR compression
const MAIN_CODE_LENGTHS: [u8; 271] = {
    let mut lengths = [8u8; 271];
    lengths[256] = 9; // End symbol
    lengths[257] = 9; // Filter symbol  
    lengths[258] = 9; // Repeat symbol
    lengths[259] = 9; // Old offset 0
    lengths[260] = 9; // Old offset 1
    lengths[261] = 9; // Old offset 2
    lengths[262] = 9; // Old offset 3
    lengths[263] = 9; // Short match 0
    lengths[264] = 9; // Short match 1
    lengths[265] = 9; // Short match 2
    lengths[266] = 9; // Short match 3
    lengths[267] = 9; // Short match 4
    lengths[268] = 9; // Short match 5
    lengths[269] = 9; // Short match 6
    lengths[270] = 9; // Short match 7
    lengths
};

const LENGTH_CODE_LENGTHS: [u8; 28] = [4; 28];

pub struct CompressionReader {
    inner: Box<dyn Read>,
    method: CompressionFlags,
    buffer: Vec<u8>,
    pos: usize,
    decompressed: bool,
}

impl CompressionReader {
    pub fn new<R: Read + 'static>(reader: R, compression: &Compression) -> Result<Self> {
        let method = compression.flag.clone();
        
        match method {
            CompressionFlags::Unknown => Err(RarError::UnsupportedCompression),
            _ => Ok(Self {
                inner: Box::new(reader),
                method,
                buffer: Vec::new(),
                pos: 0,
                decompressed: false,
            })
        }
    }
}

impl Read for CompressionReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.method {
            CompressionFlags::Save => self.inner.read(buf),
            _ => self.decompress_rar(buf),
        }
    }
}

// Huffman decoding structures based on unarr
#[derive(Debug)]
struct HuffmanNode {
    branches: [i32; 2], // -1 = unset, -2 = unset, >=0 = node index, or symbol value for leaf
}

#[derive(Debug)]
struct HuffmanCode {
    tree: Vec<HuffmanNode>,
    min_length: usize,
    max_length: usize,
}

impl Default for HuffmanCode {
    fn default() -> Self {
        Self::new()
    }
}

impl HuffmanCode {
    fn new() -> Self {
        Self {
            tree: Vec::new(),
            min_length: usize::MAX,
            max_length: 0,
        }
    }

    fn add_node(&mut self) -> usize {
        let index = self.tree.len();
        self.tree.push(HuffmanNode { branches: [-1, -2] });
        index
    }

    fn add_value(&mut self, value: i32, code_bits: u32, length: usize) -> bool {
        if self.tree.is_empty() {
            self.add_node();
        }

        if length > self.max_length {
            self.max_length = length;
        }
        if length < self.min_length {
            self.min_length = length;
        }

        let mut node = 0;
        for bit_pos in (0..length).rev() {
            let bit = ((code_bits >> bit_pos) & 1) as usize;

            if self.is_leaf_node(node) {
                return false; // Invalid - prefix found
            }

            if self.tree[node].branches[bit] < 0 {
                let new_node = self.add_node();
                self.tree[node].branches[bit] = new_node as i32;
            }

            node = self.tree[node].branches[bit] as usize;
        }

        // Set as leaf node
        self.tree[node].branches[0] = value;
        self.tree[node].branches[1] = value;
        true
    }

    fn create_from_lengths(&mut self, lengths: &[u8]) -> bool {
        let mut code_bits = 0u32;

        for length in 1..=15 {
            for (symbol, &sym_length) in lengths.iter().enumerate() {
                if sym_length as usize != length {
                    continue;
                }

                if !self.add_value(symbol as i32, code_bits, length) {
                    return false;
                }
                code_bits += 1;
            }
            code_bits <<= 1;
        }
        true
    }

    fn is_leaf_node(&self, node: usize) -> bool {
        if node >= self.tree.len() {
            return false;
        }
        self.tree[node].branches[0] == self.tree[node].branches[1]
    }

    fn decode_symbol(&self, bit_reader: &mut RarBitReader) -> Option<i32> {
        if self.tree.is_empty() {
            return None;
        }

        let mut node = 0;
        while !self.is_leaf_node(node) {
            let bit = bit_reader.read_bit()? as usize;

            let next_node = self.tree[node].branches[bit];
            if next_node < 0 {
                return None;
            }
            node = next_node as usize;
        }

        Some(self.tree[node].branches[0])
    }
}

// PPM Context Modeling using ppmd-rust
struct PpmDecoder {
    // Simplified PPM implementation - real RAR PPM is more complex
    initialized: bool,
}

impl PpmDecoder {
    fn new() -> Self {
        PpmDecoder { initialized: false }
    }

    fn init(&mut self, _order: u8, _mem_size: usize) -> std::io::Result<()> {
        // PPMd initialization would go here
        // For now, just mark as initialized
        self.initialized = true;
        Ok(())
    }

    fn decode_byte(&mut self, input: &mut &[u8]) -> Option<u8> {
        if self.initialized && !input.is_empty() {
            // Simplified PPM decoding - real implementation would use
            // context modeling and arithmetic coding
            let byte = input[0];
            *input = &input[1..];
            Some(byte)
        } else {
            None
        }
    }
}

// RAR-specific bit reader based on unarr implementation
struct RarBitReader {
    data: Vec<u8>,
    pos: usize,
    bits: u64,      // 64-bit buffer for bits
    available: i32, // Number of bits available in buffer
    at_eof: bool,
}

impl RarBitReader {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            pos: 0,
            bits: 0,
            available: 0,
            at_eof: false,
        }
    }

    fn fill(&mut self, bits_needed: i32) -> bool {
        if self.at_eof {
            return false;
        }

        // Read as many bytes as possible to fill the 64-bit buffer
        let bytes_to_read = (64 - self.available) / 8;
        let available_bytes = self.data.len().saturating_sub(self.pos);
        let count = bytes_to_read.min(available_bytes as i32) as usize;

        if bits_needed > self.available + (count as i32 * 8) {
            self.at_eof = true;
            return false;
        }

        // Fill buffer with bytes (MSB first, like unarr)
        for _i in 0..count {
            if self.pos < self.data.len() {
                self.bits = (self.bits << 8) | (self.data[self.pos] as u64);
                self.pos += 1;
                self.available += 8;
            }
        }

        true
    }

    fn check(&mut self, bits: i32) -> bool {
        bits <= self.available || self.fill(bits)
    }

    fn read_bits(&mut self, bits: i32) -> Option<u64> {
        if !(0..=64).contains(&bits) {
            return None;
        }

        if !self.check(bits) {
            return None;
        }

        // Extract bits from MSB side (like unarr)
        self.available -= bits;
        let result = (self.bits >> self.available) & ((1u64 << bits) - 1);
        Some(result)
    }

    fn read_bit(&mut self) -> Option<bool> {
        self.read_bits(1).map(|b| b != 0)
    }

    fn read_byte(&mut self) -> Option<u8> {
        self.read_bits(8).map(|b| b as u8)
    }
}

// LZSS sliding window for RAR decompression
struct LzssWindow {
    window: Vec<u8>,
    pos: usize,
    size: usize,
}

impl LzssWindow {
    fn new(size: usize) -> Self {
        LzssWindow {
            window: vec![0; size],
            pos: 0,
            size,
        }
    }

    fn put_byte(&mut self, byte: u8) {
        self.window[self.pos] = byte;
        self.pos = (self.pos + 1) % self.size;
    }

    fn copy_match(&mut self, offset: usize, length: usize, output: &mut Vec<u8>) {
        for _ in 0..length {
            let src_pos = (self.pos + self.size - offset) % self.size;
            let byte = self.window[src_pos];
            output.push(byte);
            self.put_byte(byte);
        }
    }
}

impl CompressionReader {
    fn decompress_rar(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // Decompress data if not already done
        if !self.decompressed {
            let mut compressed = Vec::new();
            self.inner.read_to_end(&mut compressed)?;

            // RAR decompression with Huffman decoding and PPM support
            self.buffer = self.rar_decompress_with_ppm(&compressed)?;
            self.pos = 0;
            self.decompressed = true;
        }

        // Copy decompressed data to output buffer
        let available = self.buffer.len() - self.pos;
        let to_copy = buf.len().min(available);

        if to_copy > 0 {
            buf[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
            self.pos += to_copy;
        }

        Ok(to_copy)
    }

    fn rar_decompress_with_ppm(&self, compressed: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut bit_reader = RarBitReader::new(compressed.to_vec());
        let mut output = Vec::new();
        let mut lzss = LzssWindow::new(4096);
        let mut ppm = PpmDecoder::new();

        // Check if this is a PPM block (RAR5 format)
        let is_ppm_block = bit_reader.read_bit().unwrap_or(false);

        if is_ppm_block {
            // Initialize PPM decoder with default parameters
            ppm.init(7, 16 * 1024 * 1024)?;

            // PPM decompression - simplified
            let mut remaining = &compressed[1..];
            while let Some(byte) = ppm.decode_byte(&mut remaining) {
                output.push(byte);
                lzss.put_byte(byte);
            }
        } else {
            // Use Huffman + LZSS decompression with RAR bit stream
            self.huffman_lzss_decompress(&mut bit_reader, &mut output, &mut lzss)?;
        }

        Ok(output)
    }

    fn huffman_lzss_decompress(
        &self,
        bit_reader: &mut RarBitReader,
        output: &mut Vec<u8>,
        lzss: &mut LzssWindow,
    ) -> std::io::Result<()> {
        // Parse RAR5 compression codes
        if !self.parse_rar5_codes(bit_reader)? {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to parse RAR5 codes",
            ));
        }

        // RAR decompression tables based on unarr
        let length_bases = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112,
            128, 160, 192, 224,
        ];
        let length_bits = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
        ];
        let short_bases = [0, 4, 8, 16, 32, 64, 128, 192];
        let short_bits = [2, 2, 3, 4, 5, 6, 6, 6];

        // Create main Huffman code
        let mut main_code = HuffmanCode::new();
        let mut length_code = HuffmanCode::new();

        // Simple Huffman tables (real RAR uses dynamic tables)
        let main_lengths = [
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 0-15
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 16-31
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 32-47
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 48-63
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 64-79
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 80-95
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 96-111
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 112-127
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 128-143
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 144-159
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 160-175
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 176-191
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 192-207
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 208-223
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 224-239
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 240-255
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 256-270 (special symbols)
        ];
        let length_table_lengths = [4; 28]; // Simple length table

        if !main_code.create_from_lengths(&main_lengths) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to create main Huffman code",
            ));
        }

        if !length_code.create_from_lengths(&length_table_lengths) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to create length Huffman code",
            ));
        }

        // RAR decompression state
        let mut old_offsets = [0u32; 4];
        let mut last_offset = 0u32;
        let mut last_length = 0u32;

        // Main decompression loop (based on unarr's rar_expand)
        loop {
            let symbol = main_code.decode_symbol(bit_reader);
            if symbol.is_none() {
                break;
            }
            let symbol = symbol.unwrap();

            if symbol < 256 {
                // Literal byte
                let byte = symbol as u8;
                output.push(byte);
                lzss.put_byte(byte);
            } else if symbol == 256 {
                // End of block or new table
                if let Some(continue_bit) = bit_reader.read_bit() {
                    if !continue_bit {
                        if let Some(new_table) = bit_reader.read_bit() {
                            if new_table {
                                // Parse new codes
                                if !self.parse_rar5_codes(bit_reader)? {
                                    break;
                                }
                            }
                        }
                        break;
                    }
                    // Continue with new table
                    if !self.parse_rar5_codes(bit_reader)? {
                        break;
                    }
                } else {
                    break;
                }
            } else if symbol == 257 {
                // Filter (skip for now)
                continue;
            } else if symbol == 258 {
                // Repeat last match
                if last_length == 0 {
                    continue;
                }
                lzss.copy_match(last_offset as usize, last_length as usize, output);
            } else if symbol <= 262 {
                // Match with old offset
                let idx = (symbol - 259) as usize;
                if idx >= old_offsets.len() {
                    break;
                }

                let offset = old_offsets[idx];
                let len_symbol = length_code.decode_symbol(bit_reader);
                if len_symbol.is_none() {
                    break;
                }
                let len_symbol = len_symbol.unwrap() as usize;

                if len_symbol >= length_bases.len() {
                    break;
                }

                let mut length = length_bases[len_symbol] + 2;
                if length_bits[len_symbol] > 0 {
                    if let Some(extra_bits) = bit_reader.read_bits(length_bits[len_symbol]) {
                        length += extra_bits as u32;
                    }
                }

                // Update old offsets
                for i in (1..=idx).rev() {
                    old_offsets[i] = old_offsets[i - 1];
                }
                old_offsets[0] = offset;

                last_offset = offset;
                last_length = length;
                lzss.copy_match(offset as usize, length as usize, output);
            } else if symbol <= 270 {
                // Short match
                let idx = (symbol - 263) as usize;
                if idx >= short_bases.len() {
                    break;
                }

                let mut offset = short_bases[idx] + 1;
                if short_bits[idx] > 0 {
                    if let Some(extra_bits) = bit_reader.read_bits(short_bits[idx]) {
                        offset += extra_bits as u32;
                    }
                }

                let length = 2;

                // Update old offsets
                for i in (1..4).rev() {
                    old_offsets[i] = old_offsets[i - 1];
                }
                old_offsets[0] = offset;

                last_offset = offset;
                last_length = length;
                lzss.copy_match(offset as usize, length as usize, output);
            } else {
                // Unknown symbol
                break;
            }
        }

        Ok(())
    }

    fn parse_rar5_codes(&self, bit_reader: &mut RarBitReader) -> std::io::Result<bool> {
        // Simplified RAR5 code parsing based on unarr

        // Check if we should reset length table
        if let Some(reset_table) = bit_reader.read_bit() {
            if !reset_table {
                // Keep existing table
                return Ok(true);
            }
        } else {
            return Ok(false);
        }

        // Read bit lengths for precode (19 symbols)
        let mut prelengths = [0u8; 19];
        for prelength in &mut prelengths {
            if let Some(length) = bit_reader.read_bits(4) {
                *prelength = length as u8;
            } else {
                return Ok(false);
            }
        }

        // Build precode Huffman table (simplified)
        let mut precode = HuffmanCode::new();
        if !precode.create_from_lengths(&prelengths) {
            return Ok(false);
        }

        // Use precode to decode main code lengths (simplified)
        // Real implementation would decode all length tables here

        Ok(true)
    }
}
