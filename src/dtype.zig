//! Internal datatypes used by zcompute

const std = @import("std");

/// Comptime-sized ring buffer
pub fn RingBuffer(comptime T: type, comptime size: usize) type {
    return struct {
        data: [size]T = undefined,
        start: usize = 0,
        len: usize = 0,

        const Self = @This();

        pub fn push(self: *Self, v: T) !void {
            if (self.len >= size) return error.OutOfMemory;
            self.data[(self.start + self.len) % size] = v;
            self.len += 1;
        }

        /// Returns the next item from the buffer and advances the pointer.
        /// If no items are left, returns null.
        pub fn pop(self: *Self) ?T {
            if (self.len == 0) return null;
            const v = self.data[self.start];
            self.start = (self.start + 1) % size;
            self.len -= 1;
            return v;
        }
    };
}
