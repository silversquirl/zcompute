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

    const Shader = zc.Shader(&.{
        zc.uniformBuffer("count", 0, zc.Buffer(u32)),
        zc.storageBuffer("in", 1, zc.Buffer(f32)),
        zc.storageBuffer("out", 2, zc.Buffer(f32)),
    });
    var shad = try Shader.initBytes(&ctx, @embedFile("test/copy.spv"));
    defer shad.deinit();

    try shad.exec(0, .{ .x = count }, .{
        .count = count_buf,
        .in = in,
        .out = out,
    });
    try shad.wait();

    {
        const data = try out.map();
        defer out.unmap();
        for (data) |v, i| {
            try std.testing.expectEqual(@intToFloat(f32, i), v);
        }
    }
}

test "push constants" {
    var ctx = try zc.Context.init(allocator);
    defer ctx.deinit();

    const count = 16;

    const out = try zc.Buffer([4]f32).init(&ctx, count, .{ .map = true, .storage = true });
    defer out.deinit();

    const Shader = zc.Shader(&.{
        zc.pushConstant("data", 0, [4]f32),
        zc.storageBuffer("out", 0, zc.Buffer([4]f32)),
    });
    var shad = try Shader.initBytes(&ctx, @embedFile("test/push-constants.spv"));
    defer shad.deinit();

    try shad.exec(0, .{ .x = count }, .{
        .data = .{ 0, 1, 2, 3 },
        .out = out,
    });
    try shad.wait();

    {
        const data = try out.map();
        defer out.unmap();
        for (data) |v| {
            try std.testing.expectEqual([4]f32{ 0, 1, 2, 3 }, v);
        }
    }
}
