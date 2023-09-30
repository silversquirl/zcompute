const std = @import("std");

pub fn build(b: *std.Build) void {
    const vk = b.dependency("vulkan_zig", .{
        .registry = @as([]const u8, b.pathFromRoot("vk.xml")),
    }).module("vulkan-zig");

    _ = b.addModule("zcompute", .{
        .source_file = .{ .path = "src/zcompute.zig" },
        .dependencies = &.{
            .{ .name = "vk", .module = vk },
        },
    });

    const test_step = b.addTest(.{
        .root_source_file = .{ .path = "src/test.zig" },
    });
    test_step.addModule("vk", vk);
    test_step.linkLibC();
    b.step("test", "Run library tests").dependOn(&test_step.step);
}
