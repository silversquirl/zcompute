//! Tests for zcompute

const std = @import("std");
const zc = @import("zcompute.zig");

var allocator_instance = std.heap.GeneralPurposeAllocator(.{
    .stack_trace_frames = 0, // Vulkan does some weird shit
}){};
const allocator = &allocator_instance.allocator;

test "initialization" {
    var ctx = try zc.Context.init(allocator);
    ctx.deinit();
}

test "alloc/free" {
    var ctx = try zc.Context.init(allocator);
    defer ctx.deinit();

    const mem = try ctx.alloc(100, .{});
    defer ctx.free(mem);
}

test "buffer creation/deletion" {
    var ctx = try zc.Context.init(allocator);
    defer ctx.deinit();

    const buf = try zc.Buffer([2]f32).init(&ctx, 6);
    defer buf.deinit();
}
