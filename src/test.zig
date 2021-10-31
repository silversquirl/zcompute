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

test "buffer mapping" {
    var ctx = try zc.Context.init(allocator);
    defer ctx.deinit();

    const buf = try zc.Buffer([2]f32).init(&ctx, 6, .{ .map = true, .uniform = true });
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

test "compute shader" {
    var ctx = try zc.Context.init(allocator);
    defer ctx.deinit();

    const count = 16;

    const in = try zc.Buffer(f32).init(&ctx, count, .{ .map = true, .storage = true });
    defer in.deinit();
    {
        const data = try in.map();
        defer in.unmap();
        for (data) |*v, i| {
            v.* = @intToFloat(f32, i);
        }
    }

    const out = try zc.Buffer(f32).init(&ctx, count, .{ .map = true, .storage = true });
    defer out.deinit();

    const count_buf = try zc.Buffer(u32).init(&ctx, 1, .{ .map = true, .uniform = true });
    defer count_buf.deinit();
    {
        const data = try count_buf.map();
        defer count_buf.unmap();
        data[0] = count;
    }

    const Shader = zc.Shader(&[_]zc.ShaderBinding{
        .{ "count", 0, .uniform, zc.Buffer(u32) },
        .{ "in", 1, .storage, zc.Buffer(f32) },
        .{ "out", 2, .storage, zc.Buffer(f32) },
    });
    var shad = try Shader.initBytes(&ctx, @embedFile("test/copy.spv"));
    defer shad.deinit();

    try shad.exec(0, .{ .direct = .{ count, 1, 1 } }, .{
        .count = count_buf,
        .in = in,
        .out = out,
    });
    _ = try shad.wait(null);

    {
        const data = try out.map();
        defer out.unmap();
        for (data) |v, i| {
            try std.testing.expectEqual(@intToFloat(f32, i), v);
        }
    }
}
