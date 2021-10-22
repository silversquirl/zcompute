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

    const buf = try zc.Buffer([2]f32).init(&ctx, 6, .{});
    defer buf.deinit();
}

test "buffer mapping" {
    var ctx = try zc.Context.init(allocator);
    defer ctx.deinit();

    const buf = try zc.Buffer([2]f32).init(&ctx, 6, .{ .map = true });
    defer buf.deinit();

    {
        const map = try buf.map();
        defer buf.unmap();

        map[0] = .{ 0, 1 };
        map[1] = .{ 2, 3 };
        map[2] = .{ 4, 5 };
        map[3] = .{ 6, 7 };
        map[4] = .{ 8, 9 };
        map[5] = .{ 10, 11 };
    }

    {
        const map = try buf.map();
        defer buf.unmap();

        try std.testing.expectEqualSlices([2]f32, &.{
            .{ 0, 1 },
            .{ 2, 3 },
            .{ 4, 5 },
            .{ 6, 7 },
            .{ 8, 9 },
            .{ 10, 11 },
        }, map);
    }
}
